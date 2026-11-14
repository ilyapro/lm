#!/usr/bin/env python3
"""Back up the live Living Memory SQLite database to one private S3 object."""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import uuid
import zlib
from datetime import datetime, timezone
from urllib.parse import quote
from xml.etree import ElementTree


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


def exception_chain(error: BaseException) -> list[dict]:
    """Only local types and bounded integers, never exception text or args."""
    chain, seen = [], set()
    while error is not None and id(error) not in seen and len(chain) < 8:
        seen.add(id(error))
        name = type(error).__name__
        item = {"class": name if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,63}", name) else "unclassified"}
        for key, low, high in (("errno", -(2**31), 2**31 - 1), ("status", 100, 599)):
            value = getattr(error, key, None)
            if type(value) is int and low <= value <= high:
                item[key] = value
        chain.append(item)
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return chain


def checked_chain(value: object) -> list[dict]:
    if not isinstance(value, list) or len(value) > 8:
        return []
    for item in value:
        if (not isinstance(item, dict) or not {"class"} <= item.keys() <= {"class", "errno", "status"}
                or not isinstance(item["class"], str)
                or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,63}", item["class"])):
            return []
        for key, low, high in (("errno", -(2**31), 2**31 - 1), ("status", 100, 599)):
            if key in item and (type(item[key]) is not int or not low <= item[key] <= high):
                return []
    return value


class S3Failure(RuntimeError):
    def __init__(self, operation: str, chain: object, exit_code: int | None = None):
        chain = checked_chain(chain)
        self.transport = {"operation": operation, "chain": chain}
        if exit_code is not None:
            self.transport["exit_code"] = exit_code
        detail = " -> ".join(
            item["class"] + "".join(f" {key}={item[key]}" for key in ("errno", "status") if key in item)
            for item in chain) or "cause unavailable"
        code = f"exit {exit_code}; " if exit_code is not None else ""
        super().__init__(f"s3cmd {operation} failed ({code}{detail})")


def error_record(error: Exception) -> dict:
    record = {"at": now(), "message": str(error)}
    if isinstance(error, S3Failure):
        record["transport"] = error.transport
    return record


def transport(descriptor: str, *args: str) -> None:
    """Execute installed s3cmd, observing errors before its CLI discards them."""
    from contextlib import redirect_stderr
    import logging
    import runpy

    # This is a dedicated subprocess. No library diagnostics reach its stderr.
    logging.disable(logging.CRITICAL)
    with os.fdopen(int(descriptor), "w", closefd=False) as diagnostic, open(os.devnull, "w") as quiet:
        def record(chain):
            diagnostic.seek(0)
            diagnostic.truncate()
            diagnostic.write(json.dumps(chain))
            diagnostic.flush()

        with redirect_stderr(quiet):
            try:
                from S3.S3 import S3

                depth = 0

                def observe(method):
                    def call(*positional, **keywords):
                        nonlocal depth
                        if depth == 0:
                            record([])
                        depth += 1
                        try:
                            return method(*positional, **keywords)
                        except Exception as error:
                            if depth == 1:
                                record(exception_chain(error))
                            raise
                        finally:
                            depth -= 1
                    return call

                for name in ("send_request", "send_file", "send_file_multipart", "recv_file"):
                    setattr(S3, name, observe(getattr(S3, name)))
                sys.argv = ["/usr/bin/s3cmd", *args]
                runpy.run_path("/usr/bin/s3cmd", run_name="__main__")
            except SystemExit as error:
                # CLI handlers can also retain a cause on their SystemExit.
                if error.code and isinstance(error.__context__, Exception):
                    record(exception_chain(error.__context__))
                raise
            except Exception as error:
                record(exception_chain(error))
                raise SystemExit(1) from None


def s3(*args: str) -> str:
    with tempfile.TemporaryFile(mode="w+") as diagnostic:
        try:
            result = subprocess.run(
                ["/usr/bin/python3", "-c",
                 "import runpy, sys; runpy.run_path(sys.argv[1])['transport'](*sys.argv[2:])",
                 str(Path(__file__).resolve()), str(diagnostic.fileno()),
                 "-c", str(CONFIG), "--no-progress", "--max-retries=0", *args],
                pass_fds=(diagnostic.fileno(),), capture_output=True, text=True,
                timeout=TRANSFER_TIMEOUT if args[0] in {"get", "put", "cp"} else READ_TIMEOUT,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise S3Failure(args[0], exception_chain(error)) from None
        if result.returncode:
            diagnostic.seek(0)
            try:
                chain = json.loads(diagnostic.read(4096))
            except ValueError:
                chain = []
            raise S3Failure(args[0], chain, result.returncode)
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


def metadata(config_path: str, target: str, content: str) -> None:
    """Read required facts in a bounded subprocess using s3cmd's own transport.

    Do not use bucket_info/cmd_info: they turn failed versioning/policy reads
    into display values indistinguishable from confirmed absence.
    """
    import logging
    import socket

    # Library diagnostics may contain signed headers, credentials or raw bodies.
    logging.disable(logging.CRITICAL)
    operation = "config"
    try:
        from S3.Config import Config
        from S3.Exceptions import S3Error
        from S3.S3 import S3
        from S3.S3Uri import S3Uri

        config = Config(config_path)
        config.max_retries = 0
        socket.setdefaulttimeout(config.socket_timeout)
        client = S3(config)
        uri = S3Uri(target)
        fields = {}
        if not uri.has_object():
            operation = "versioning"
            response = client.send_request(client.create_request(
                "BUCKET_LIST", bucket=uri.bucket(), uri_params={"versioning": None}))
            root = ElementTree.fromstring(response["data"])
            # Only a successful, empty VersioningConfiguration is unversioned.
            # An empty body, unknown XML, Status (even empty) or MFADelete is not.
            if (response["status"] != 200 or root.tag not in {
                    "VersioningConfiguration",
                    "{http://s3.amazonaws.com/doc/2006-03-01/}VersioningConfiguration",
                } or root.attrib or len(root) or (root.text or "").strip()):
                raise ValueError("versioning is not confirmed empty")
            fields["Versioning"] = "none"

        operation = "acl"
        acl = client.get_acl(uri)
        grants = acl.getGrantList()
        if ([f"{g['grantee']}: {g['permission']}" for g in grants] != PRIVATE_ACL
                or not acl.owner_id or len(acl.grantees) != 1
                or acl.grantees[0].xsi_type != "CanonicalUser"
                or acl.grantees[0].tag != "ID"
                or acl.grantees[0].name != acl.owner_id):
            raise ValueError("ACL is not owner-only FULL_CONTROL")

        operation = "policy"
        try:
            client.get_policy(uri)
        except S3Error as error:
            if error.status != 404 or error.code != "NoSuchBucketPolicy":
                raise
        else:
            # Even an empty/invalid policy body is not authoritative absence.
            raise ValueError("bucket policy is not confirmed absent")

        if uri.has_object() and content == "1":
            operation = "head"
            response = client.object_info(uri)
            headers = response["headers"]
            length = headers.get("content-length", "")
            md5 = response.get("s3cmd-attrs", {}).get("md5")
            if md5 is None:
                md5 = headers.get("etag", "").strip('"\'')
            # s3cmd's original MD5 takes precedence over an opaque multipart
            # ETag. Neither a multipart ETag nor malformed metadata is an MD5.
            if (response["status"] != 200 or not re.fullmatch(r"[0-9]+", length)
                    or not re.fullmatch(r"[0-9a-fA-F]{32}", md5)):
                raise ValueError("invalid content metadata")
            fields.update({"File size": str(int(length)), "MD5 sum": md5.lower()})
        result = {"fields": fields}
    except Exception as error:
        result = {"error": operation, "chain": exception_chain(error)}
    print(json.dumps(result))


def info(target: str, *, content: bool = True) -> dict[str, str]:
    # Use the interpreter owning /usr/bin/s3cmd's installed S3 package. Keeping
    # this in a subprocess retains the whole-read deadline, including redirects.
    try:
        result = subprocess.run(
            ["/usr/bin/python3", "-c",
             "import runpy, sys; runpy.run_path(sys.argv[1])['metadata'](*sys.argv[2:])",
             str(Path(__file__).resolve()), str(CONFIG), target, "1" if content else "0"],
            capture_output=True, text=True, timeout=READ_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise S3Failure("metadata", exception_chain(error)) from None
    if result.returncode:
        raise S3Failure("metadata", [], result.returncode)
    try:
        response = json.loads(result.stdout)
        if "error" in response:
            operation = response["error"]
            if operation not in {"config", "versioning", "acl", "policy", "head"}:
                raise ValueError
            raise S3Failure(f"metadata {operation}", response.get("chain"))
        return response["fields"]
    except (ValueError, KeyError, TypeError):
        raise RuntimeError("invalid S3 metadata response") from None


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
                    cleanup_error=error_record(error))
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
    info(target, content=False)  # Fresh ACL/policy check at each verification.
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
    # Metadata uses the original content MD5 from s3cmd-attrs, including
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


def matches_previous(observed: dict, archive: Path, previous: dict) -> bool:
    if "archive_sha256" in previous:
        return observed == previous
    # Before safety verification, only the original HEAD metadata is durable.
    md5 = hashlib.md5(usedforsecurity=False)
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            md5.update(block)
    return (str(archive.stat().st_size) == previous.get("File size")
            and md5.hexdigest() == previous.get("MD5 sum"))


def record_success(transaction: dict) -> None:
    transaction.update(phase="verified", pending=None)
    save_status(transaction=transaction, result="verified", last_success=now(),
                last_error=None, recovery_error=None, rollback=None, rollback_error=None,
                **transaction["expected"])


def recover(transaction: dict) -> int | None:
    """Reconcile under backup.lock; None admits one fresh scheduled attempt."""
    phase, pending = transaction["phase"], transaction["pending"]
    safety, previous, expected = (transaction[key] for key in ("safety", "previous", "expected"))
    completed = phase in {"verified", "rolled_back", "unchanged"}
    if phase not in {"preparing", "safety_copied", "safety_verified", "uploaded",
                     "replaced", "restored", "verified", "rolled_back", "unchanged"}:
        raise RuntimeError("unrecognized backup recovery phase")
    if safety is not None and (not isinstance(safety, str) or not re.fullmatch(
            re.escape(f"s3://{BUCKET}/.previous-") + r"[0-9a-f]{32}\.sqlite3\.gz", safety)):
        raise RuntimeError("unrecognized backup safety target")
    if pending:
        allowed = {("put", STAGING), ("cp", FINAL)}
        if safety:
            allowed.add(("cp", safety))
        cleanup = {("del", STAGING), ("abortmp", STAGING)}
        if safety:
            cleanup.add(("del", safety))
        if (pending.get("operation"), pending.get("target")) not in (cleanup if completed else allowed):
            raise RuntimeError("unrecognized pending backup operation")
        if pending["operation"] == "abortmp" and not pending.get("upload_id"):
            raise RuntimeError("missing pending multipart upload identity")

    check_config()
    if info(f"s3://{BUCKET}").get("Versioning") != "none":
        raise RuntimeError("S3 bucket versioning must be disabled")
    keys, uploads = objects(), multipart()
    if keys - {FINAL, STAGING, safety} or any(key != STAGING for key, _ in uploads):
        raise RuntimeError("unowned S3 objects or uploads; preserving recovery state")

    if completed:
        # A committed success is never rolled back because cleanup was interrupted.
        if (FINAL in keys) != bool(phase == "verified" or previous):
            raise RuntimeError("unexpected FINAL inventory during cleanup recovery")
        if pending and pending["operation"] == "del":
            checkpoint = "staging_deleted" if pending["target"] == STAGING else "safety_deleted"
            transaction[checkpoint] = pending["target"] not in keys
        transaction["multipart_clean"] = not uploads
        transaction["pending"] = None
        save_status(transaction=transaction, recovery_error=None)
        return finish(transaction)

    with tempfile.TemporaryDirectory(prefix="recovery-", dir=STATE_DIR) as directory:
        work = Path(directory)
        observed = verify_download(FINAL, work, "current") if FINAL in keys else None
        if expected and observed == expected:
            record_success(transaction)
            return finish(transaction)
        if previous:
            if observed is None or not matches_previous(observed, work / "current.sqlite3.gz", previous):
                raise RuntimeError("FINAL matches neither recorded backup; preserving all copies")
        elif observed is not None:
            raise RuntimeError("unexpected FINAL during first-backup recovery")
        if safety in keys:
            retained = verify_download(safety, work, "retained")
            if not previous or not matches_previous(retained, work / "retained.sqlite3.gz", previous):
                raise RuntimeError("retained safety does not match previous backup")
        elif previous and phase not in {"preparing", "safety_copied"}:
            raise RuntimeError("retained safety is missing; preserving recovery state")

        if STAGING in keys:
            if not expected or (previous and safety not in keys):
                raise RuntimeError("staging lacks recorded checksums or verified safety")
            verify_download(STAGING, work, "staging", expected)
            # A fresh read established the old FINAL and the exact complete upload.
            transaction.update(pending=None, phase="uploaded")
            save_status(transaction=transaction, recovery_error=None)
            mutate(transaction, "replaced", "cp", "--acl-private", STAGING, FINAL)
            verify_download(FINAL, work, "final", expected)
            record_success(transaction)
            return finish(transaction)

        # No complete upload exists. The verified old FINAL stays in place while
        # cleanup aborts only currently listed uploads for this operation's key.
        transaction.update(pending=None, phase="unchanged", staging_deleted=True,
                           safety_deleted=safety not in keys, multipart_clean=not uploads)
        save_status(transaction=transaction, result="failed", recovery_error=None)
        result = finish(transaction)
        if read_status().get("transaction") is not None:
            return result
    return None


def run() -> int:
    """Called only by main(), while holding backup.lock for the whole operation."""
    status = read_status()
    transaction = status.get("transaction")
    if transaction:
        try:
            if not transaction.get("pending") and transaction.get("phase") in {"verified", "rolled_back", "unchanged"}:
                check_config()
                result = finish(transaction)
            else:
                result = recover(transaction)
        except (OSError, sqlite3.Error, RuntimeError, ValueError, EOFError) as error:
            current = read_status()
            save_status(recovery_error=error_record(error),
                        result="verified" if current.get("result") == "verified" else "unknown")
            print(f"Living Memory backup recovery deferred: {error}", file=sys.stderr)
            return 3
        # Completed cleanup of an unsuccessful attempt admits today's snapshot.
        # A verified result resumes only its cleanup and keeps its success time.
        if result is not None and not (result == 1 and read_status().get("transaction") is None):
            return result

    try:
        before = preflight()
    except (OSError, RuntimeError) as error:
        save_status(result="failed", last_attempt=now(), cleanup="unverified",
                    last_error=error_record(error))
        raise
    transaction = {
        "phase": "preparing", "pending": None,
        "safety": f"s3://{BUCKET}/.previous-{uuid.uuid4().hex}.sqlite3.gz" if before else None,
        "previous": before, "expected": None,
    }
    save_status(transaction=transaction, result="in_progress", cleanup="pending",
                last_attempt=now(), last_error=None, cleanup_error=None, recovery_error=None,
                rollback=None, rollback_error=None)
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
                        last_error=error_record(error), cleanup="unverified")
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
                                rollback_error=error_record(restore_error), cleanup="unverified")
                    print("Living Memory rollback unverified; safety retained", file=sys.stderr)
                    return 3 if transaction["pending"] else 1
            else:
                # FINAL was never replaced, so removing our temporary copies is safe.
                transaction["phase"] = "unchanged"
                save_status(transaction=transaction)
            return finish(transaction)

        # Commit the verified result BEFORE cleanup. No later exception enters
        # rollback, erases success or authorizes a consumer to repeat this work.
        record_success(transaction)
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
                                cleanup_error=error_record(error))
                    return 3 if pending else 2
                if pending:
                    return 3
            except (OSError, RuntimeError, ValueError):
                pass  # Unreadable status is retained, never reset to an empty run.
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
