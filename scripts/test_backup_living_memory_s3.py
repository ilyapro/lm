"""Private, offline command-path regression. Run with pytest on this file only."""

import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import runpy
import socket
import sqlite3
import subprocess
import sys
import tempfile

import pytest


SCRIPT = Path(__file__).with_name("backup_living_memory_s3.py")
BUCKET = "s3://lm-private-backup-sfx"
FINAL = BUCKET + "/global.sqlite3.gz"
STAGING = BUCKET + "/.incoming.sqlite3.gz"


class FakeS3:
    def __init__(self, home):
        self.home = home
        self.objects = {}
        self.calls = []
        self.hook = lambda operation, args: None
        self.public = set()
        self.versioning = "none"
        self.policy = "none"
        self.uploads = []

    def __call__(self, command, **kwargs):
        assert command[:3] == ["/usr/bin/s3cmd", "-c", str(self.home / ".config/living-memory/s3cmd-backup.conf")]
        args = [arg for arg in command[3:] if not arg.startswith("--")]
        operation, *args = args
        # Every remote read/mutation belongs to the actual command's lock.
        with (self.home / ".local/state/living-memory-backup/backup.lock").open("a") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert "--max-retries=0" in command
        assert 0 < kwargs["timeout"] <= 3600
        self.calls.append((operation, tuple(args)))
        override = self.hook(operation, args)
        if override is not None:
            return override
        output = ""
        if operation == "info":
            target = args[0]
            output = "   Policy:    " + self.policy + "\n   ACL:       sfx: FULL_CONTROL\n"
            if target in self.public:
                output += "   ACL:       *anon*: READ\n"
            if target.rstrip("/") == BUCKET:
                output += "   Versioning:" + self.versioning + "\n"
            else:
                payload = self.objects[target]
                output += f"   File size: {len(payload)}\n   MD5 sum: {hashlib.md5(payload).hexdigest()}\n"
        elif operation == "ls":
            output = "".join(f"2026-01-01 00:00 {len(value)} {key}\n" for key, value in sorted(self.objects.items()))
        elif operation == "multipart":
            output = BUCKET + "/\nInitiated\tPath\tId\n"
            output += "".join(f"date\t{key}\t{identifier}\n" for key, identifier in self.uploads)
        elif operation == "put":
            assert "--acl-private" in command
            self.objects[args[1]] = Path(args[0]).read_bytes()
        elif operation == "cp":
            assert "--acl-private" in command
            self.objects[args[1]] = self.objects[args[0]]
        elif operation == "get":
            Path(args[1]).write_bytes(self.objects[args[0]])
        elif operation == "del":
            self.objects.pop(args[0], None)
        elif operation == "abortmp":
            self.uploads.remove(tuple(args))
        else:
            raise AssertionError(f"unexpected command: {command}")
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")


@pytest.fixture
def backup(tmp_path, monkeypatch):
    home = tmp_path / "home"
    database = home / ".local/share/living-memory/global.sqlite3"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE example (value TEXT)")
        connection.execute("INSERT INTO example VALUES ('fresh')")
    previous = tmp_path / "previous.sqlite3"
    with sqlite3.connect(previous) as connection:
        connection.execute("CREATE TABLE example (value TEXT)")
        connection.execute("INSERT INTO example VALUES ('previous')")
    old = gzip.compress(previous.read_bytes(), mtime=0)
    config = home / ".config/living-memory/s3cmd-backup.conf"
    config.parent.mkdir(parents=True)
    config.write_text("FAKE: no credentials\n")
    config.chmod(0o600)
    fake = FakeS3(home)
    fake.objects[FINAL] = old
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network forbidden"))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    original_connect = sqlite3.connect

    def isolated_connect(path, *args, **kwargs):
        assert str(tmp_path) in str(path), "non-fixture SQLite access forbidden"
        return original_connect(path, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", isolated_connect)
    state = home / ".local/state/living-memory-backup"
    original_umask = os.umask(0o077)
    os.umask(original_umask)

    def invoke():
        before = database.read_bytes()
        with pytest.raises(SystemExit) as stopped:
            runpy.run_path(str(SCRIPT), run_name="__main__")
        assert database.read_bytes() == before
        return stopped.value.code

    fake.invoke = invoke
    fake.old = old
    fake.database = database
    fake.state = state
    fake.status = lambda: json.loads((state / "status.json").read_text())

    def fresh_bytes():
        payload = gzip.decompress(fake.objects[FINAL])
        restored = tmp_path / "assertion.sqlite3"
        restored.write_bytes(payload)
        with sqlite3.connect(restored) as connection:
            assert connection.execute("SELECT value FROM example").fetchall() == [("fresh",)]
            assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        return payload

    fake.fresh_bytes = fresh_bytes
    yield fake
    os.umask(original_umask)


def test_success_command(backup):
    def committed_before_cleanup(operation, args):
        if operation == "del" and args == [STAGING]:
            assert backup.status()["result"] == "verified"
            assert backup.status()["last_success"]
            assert backup.old in backup.objects.values()
        if (backup.state / "status.json").exists():
            assert backup.status().get("cleanup") != "complete"

    backup.hook = committed_before_cleanup
    assert backup.invoke() == 0
    assert set(backup.objects) == {FINAL}
    assert backup.objects[FINAL] != backup.old
    assert backup.status()["sha256"] == hashlib.sha256(backup.fresh_bytes()).hexdigest()
    assert backup.status()["result"] == "verified"
    assert backup.status()["cleanup"] == "complete"
    assert backup.status()["transaction"] is None
    # One initial admission read, one cleanup read; never a consumer post-success query.
    assert sum(op == "multipart" for op, _ in backup.calls) == 2
    assert sum(op == "get" and args[0] == FINAL for op, args in backup.calls) == 1
    assert sum(op == "put" for op, _ in backup.calls) == 1
    assert (backup.state / "status.json").stat().st_mode & 0o077 == 0


def test_late_multipart_timeout_keeps_verified_result(backup):
    def fail_cleanup(operation, args):
        if operation == "multipart" and backup.objects.get(FINAL) != backup.old:
            raise subprocess.TimeoutExpired("fake multipart", 120)

    backup.hook = fail_cleanup
    assert backup.invoke() == 2
    backup.fresh_bytes()
    status = backup.status()
    assert status["result"] == "verified"
    assert status["last_success"]
    assert status["cleanup"] == "unverified"
    assert backup.old in backup.objects.values()


def seed_success(backup):
    backup.state.mkdir(parents=True, exist_ok=True)
    status = {"last_success": "2000-01-01T00:00:00Z", "sha256": "prior-success", "bytes": 8192}
    (backup.state / "status.json").write_text(json.dumps(status))
    return status


def test_error_before_replacement_preserves_old_bytes(backup):
    previous = seed_success(backup)

    def corrupt_staging(operation, args):
        if operation == "get" and args[0] == STAGING:
            Path(args[1]).write_bytes(b"broken download")
            return subprocess.CompletedProcess([], 0, "", "")

    backup.hook = corrupt_staging
    assert backup.invoke() == 1
    assert backup.objects == {FINAL: backup.old}
    assert not any(op == "cp" and args[-1] == FINAL for op, args in backup.calls)
    status = backup.status()
    assert status["result"] == "failed"
    assert status["last_success"] == previous["last_success"]
    assert status["sha256"] == previous["sha256"]


@pytest.mark.parametrize("failure", ["corrupt", "timeout", "acl"])
def test_final_verification_failure_restores_previous(backup, failure):
    previous = seed_success(backup)
    failed = False

    def fail_new_final(operation, args):
        nonlocal failed
        if not failed and args[0] == FINAL and backup.objects[FINAL] != backup.old:
            if failure == "acl" and operation == "info":
                failed = True
                return subprocess.CompletedProcess([], 0, "Policy: none\nACL: *anon*: READ\n", "")
            if operation == "get":
                failed = True
                if failure == "timeout":
                    raise subprocess.TimeoutExpired("FAKE SECRET", 120)
                backup.objects[FINAL] = b"corrupt new FINAL"

    backup.hook = fail_new_final
    assert backup.invoke() == 1
    assert failed
    assert backup.objects == {FINAL: backup.old}
    status = backup.status()
    assert status["rollback"] == "verified"
    assert status["result"] == "failed"
    assert status["last_success"] == previous["last_success"]
    assert status["sha256"] == previous["sha256"]
    assert "FAKE SECRET" not in json.dumps(status)
    assert sum(op == "cp" and args[-1] == FINAL for op, args in backup.calls) == 2


@pytest.mark.parametrize("failure", ["mutation_timeout", "verification"])
def test_rollback_failure_retains_safety_and_blocks_restart(backup, failure):
    restoring = False

    def fail_restore(operation, args):
        nonlocal restoring
        if operation == "get" and args[0] == FINAL:
            if restoring:
                Path(args[1]).write_bytes(b"bad restored download")
                return subprocess.CompletedProcess([], 0, "", "")
            backup.objects[FINAL] = b"bad new FINAL"
        if operation == "cp" and args[-1] == FINAL and args[0] != STAGING:
            restoring = True
            if failure == "mutation_timeout":
                raise subprocess.TimeoutExpired("fake restore", 120)

    backup.hook = fail_restore
    assert backup.invoke() == (3 if failure == "mutation_timeout" else 1)
    status = backup.status()
    safety = status["transaction"]["safety"]
    assert backup.objects[safety] == backup.old
    assert STAGING in backup.objects
    assert status["rollback"] == ("unknown" if failure == "mutation_timeout" else "failed")
    assert not any(op == "del" for op, _ in backup.calls)
    saved = dict(backup.objects)
    calls = list(backup.calls)
    assert backup.invoke() == 3
    assert backup.objects == saved
    assert backup.calls == calls


@pytest.mark.parametrize("operation", ["safety", "upload", "replace"])
@pytest.mark.parametrize("applied", [False, True])
def test_unknown_mutations_are_not_replayed(backup, operation, applied):
    def lost_response(verb, args):
        matched = ((operation == "safety" and verb == "cp" and args[0] == FINAL)
                   or (operation == "upload" and verb == "put")
                   or (operation == "replace" and verb == "cp" and args[0] == STAGING))
        if matched:
            # Durable intent exists before even a lost response can apply the effect.
            assert backup.status()["transaction"]["pending"]["operation"] == verb
            if applied:
                backup.objects[args[-1]] = Path(args[0]).read_bytes() if verb == "put" else backup.objects[args[0]]
            raise subprocess.TimeoutExpired("fake mutation", 120)

    backup.hook = lost_response
    assert backup.invoke() == 3
    saved = dict(backup.objects)
    calls = list(backup.calls)
    safety = backup.status()["transaction"]["safety"]
    assert backup.status()["result"] == "unknown"
    if operation != "safety" or applied:
        assert backup.objects[safety] == backup.old
    assert backup.invoke() == 3
    assert backup.calls == calls
    assert backup.objects == saved
    assert backup.status()["transaction"]["safety"] == safety


@pytest.mark.parametrize("target", ["staging", "safety"])
def test_unknown_cleanup_preserves_success_and_does_not_repeat_delete(backup, target):
    def delete_then_timeout(operation, args):
        if operation == "del" and (args[0] == STAGING) == (target == "staging"):
            backup.objects.pop(args[0])
            raise subprocess.TimeoutExpired("fake delete", 120)

    backup.hook = delete_then_timeout
    assert backup.invoke() == 3
    status = backup.status()
    backup.fresh_bytes()
    assert status["result"] == "verified"
    assert status["cleanup"] == "unknown"
    assert status["last_error"] is None
    if target == "staging":
        assert backup.objects[status["transaction"]["safety"]] == backup.old
    calls = list(backup.calls)
    assert backup.invoke() == 3
    assert backup.calls == calls
    assert backup.status()["last_success"] == status["last_success"]


@pytest.mark.parametrize("operation", ["multipart", "ls"])
def test_cleanup_read_timeout_resumes_only_cleanup(backup, operation):
    def late_timeout(verb, args):
        if verb == operation and backup.objects.get(FINAL) != backup.old:
            raise subprocess.TimeoutExpired("fake cleanup read", 120)

    backup.hook = late_timeout
    assert backup.invoke() == 2
    status = backup.status()
    fresh = backup.objects[FINAL]
    assert status["result"] == "verified"
    assert status["cleanup"] == "unverified"
    assert status["last_error"] is None
    if operation == "multipart":
        assert backup.objects[status["transaction"]["safety"]] == backup.old
    backup.hook = lambda *args: None
    backup.calls.clear()
    assert backup.invoke() == 0
    assert backup.objects == {FINAL: fresh}
    assert backup.status()["last_success"] == status["last_success"]
    assert not any(op in {"get", "put", "cp", "info"} for op, _ in backup.calls)
    assert sum(op == "multipart" for op, _ in backup.calls) == (1 if operation == "multipart" else 0)
    assert sum(op == "ls" for op, _ in backup.calls) == 1
    assert not any(op == "del" and args[0] == STAGING for op, args in backup.calls)


def test_no_previous_final_is_a_valid_first_backup(backup):
    backup.objects.clear()
    assert backup.invoke() == 0
    assert set(backup.objects) == {FINAL}
    backup.fresh_bytes()
    assert not any(op == "cp" and args[0] == FINAL for op, args in backup.calls)


def test_first_backup_final_failure_keeps_checked_staging(backup):
    backup.objects.clear()

    def corrupt_final(operation, args):
        if operation == "get" and args[0] == FINAL:
            backup.objects[FINAL] = b"corrupt"

    backup.hook = corrupt_final
    assert backup.invoke() == 1
    assert backup.status()["rollback"] == "unavailable"
    assert not backup.status().get("last_success")
    assert STAGING in backup.objects
    assert gzip.decompress(backup.objects[STAGING]).startswith(b"SQLite format 3")
    assert not any(op == "del" for op, _ in backup.calls)


def test_busy_lock_does_no_work_and_preserves_status(backup):
    previous = seed_success(backup)
    with (backup.state / "backup.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert backup.invoke() == 1
    assert backup.calls == []
    assert backup.status() == previous
    assert backup.objects == {FINAL: backup.old}


@pytest.mark.parametrize("failure", ["config", "bucket_acl", "final_acl", "policy", "versioning", "extra_object", "multipart", "list_timeout", "head_failure", "corrupt_status"])
def test_admission_fails_closed_without_mutations(backup, failure):
    if failure == "config":
        (backup.home / ".config/living-memory/s3cmd-backup.conf").chmod(0o644)
    elif failure == "bucket_acl":
        backup.public.add(BUCKET)
    elif failure == "final_acl":
        backup.public.add(FINAL)
    elif failure == "policy":
        backup.policy = "unknown policy"
    elif failure == "versioning":
        backup.versioning = "enabled"
    elif failure == "extra_object":
        backup.objects[BUCKET + "/.previous-unowned.sqlite3.gz"] = backup.old
    elif failure == "multipart":
        backup.uploads.append((STAGING, "unowned-upload"))
    elif failure == "corrupt_status":
        backup.state.mkdir(parents=True)
        (backup.state / "status.json").write_text("{unreadable")
    else:
        def unavailable(operation, args):
            if failure == "list_timeout" and operation == "ls":
                raise subprocess.TimeoutExpired("fake listing", 120)
            if failure == "head_failure" and operation == "info" and args[0] == FINAL:
                return subprocess.CompletedProcess([], 1, "", "FAKE SECRET")
        backup.hook = unavailable
    saved = dict(backup.objects)
    assert backup.invoke() == 1
    assert backup.objects == saved
    assert not any(op in {"cp", "put", "del", "abortmp"} for op, _ in backup.calls)
    if failure == "corrupt_status":
        assert (backup.state / "status.json").read_text() == "{unreadable"


def test_arguments_never_start_a_backup(backup, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--help"])
    assert backup.invoke() == 1
    assert backup.calls == []
    assert not backup.state.exists()


@pytest.mark.parametrize("failure", ["metadata", "bytes", "acl"])
def test_safety_must_be_private_and_verified_before_upload(backup, failure):
    def bad_safety(operation, args):
        if "/.previous-" in args[0]:
            if failure == "metadata" and operation == "info":
                backup.objects[args[0]] = b"bad copy"
            if failure == "bytes" and operation == "get":
                Path(args[1]).write_bytes(b"bad download")
                return subprocess.CompletedProcess([], 0, "", "")
            if failure == "acl" and operation == "info":
                backup.public.add(args[0])

    backup.hook = bad_safety
    assert backup.invoke() == 1
    assert backup.objects == {FINAL: backup.old}
    assert not any(op == "put" or (op == "cp" and args[-1] == FINAL) for op, args in backup.calls)
    assert not backup.status().get("last_success")


def test_crash_after_mutation_retains_durable_intent(backup):
    def interrupted(operation, args):
        if operation == "cp" and args[0] == STAGING:
            backup.objects[FINAL] = backup.objects[STAGING]
            raise KeyboardInterrupt("simulated process death before response")

    backup.hook = interrupted
    with pytest.raises(KeyboardInterrupt):
        backup.invoke()
    status = backup.status()
    safety = status["transaction"]["safety"]
    assert status["transaction"]["pending"] == {"operation": "cp", "target": FINAL}
    assert backup.objects[safety] == backup.old
    saved = dict(backup.objects)
    backup.calls.clear()
    assert backup.invoke() == 3
    assert backup.calls == []
    assert backup.objects == saved
    assert backup.status()["result"] == "unknown"


@pytest.mark.parametrize("timeout", [False, True])
def test_owned_multipart_cleanup_is_not_requeried(backup, timeout):
    def incomplete_upload(operation, args):
        if operation == "get" and args[0] == FINAL:
            backup.uploads.append((STAGING, "owned-upload"))
        if operation == "abortmp" and timeout:
            assert backup.status()["transaction"]["pending"] == {
                "operation": "abortmp", "target": STAGING, "upload_id": "owned-upload"}
            backup.uploads.clear()
            raise subprocess.TimeoutExpired("fake abort", 120)

    backup.hook = incomplete_upload
    assert backup.invoke() == (3 if timeout else 0)
    assert backup.uploads == []
    backup.fresh_bytes()
    assert sum(op == "multipart" for op, _ in backup.calls) == 2
    assert sum(op == "abortmp" for op, _ in backup.calls) == 1
    assert backup.status()["result"] == "verified"
    if timeout:
        assert backup.old in backup.objects.values()
        backup.calls.clear()
        assert backup.invoke() == 3
        assert backup.calls == []


def test_empty_multipart_response_does_not_claim_cleanup(backup):
    def truncated_listing(operation, args):
        if operation == "multipart" and backup.objects[FINAL] != backup.old:
            return subprocess.CompletedProcess([], 0, "", "")

    backup.hook = truncated_listing
    assert backup.invoke() == 2
    assert backup.status()["result"] == "verified"
    assert backup.status()["cleanup"] == "unverified"
    assert backup.old in backup.objects.values()


def test_local_cleanup_failure_cannot_rollback_success(backup, monkeypatch):
    original_cleanup = tempfile.TemporaryDirectory.cleanup

    def failed_cleanup(self):
        original_cleanup(self)
        raise OSError("fake local cleanup failure")

    monkeypatch.setattr(tempfile.TemporaryDirectory, "cleanup", failed_cleanup)
    assert backup.invoke() == 2
    status = backup.status()
    assert status["result"] == "verified"
    assert status["last_success"]
    assert status["cleanup"] == "unverified"
    assert backup.old in backup.objects.values()
    backup.fresh_bytes()
    assert sum(op == "cp" and args[-1] == FINAL for op, args in backup.calls) == 1
