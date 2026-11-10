"""Private, offline command-path regression. Run with pytest on this file only."""

import fcntl
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import gzip
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import runpy
import socket
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
from S3.Config import Config
from S3.ConnMan import ConnMan
from S3.S3 import S3Request


SCRIPT = Path(__file__).with_name("backup_living_memory_s3.py")
BUCKET = "s3://lm-private-backup-sfx"
FINAL = BUCKET + "/global.sqlite3.gz"
STAGING = BUCKET + "/.incoming.sqlite3.gz"
NAMESPACE = "http://s3.amazonaws.com/doc/2006-03-01/"


class RawResponse:
    def __init__(self, status=200, data=b"", headers=None):
        self.status = status
        self.reason = "fixture response"
        self.data = data
        self.headers = headers or {}

    def getheaders(self):
        return list(self.headers.items())

    def read(self):
        return self.data


def raw_error(status, code):
    return RawResponse(status, f"<Error><Code>{code}</Code><Message>FAKE SECRET</Message></Error>".encode())


def raw_acl(grants=None):
    if grants is None:
        grants = [("fixture-owner", "sfx", "FULL_CONTROL")]
    xml = f'<AccessControlPolicy xmlns="{NAMESPACE}"><Owner><ID>fixture-owner</ID><DisplayName>sfx</DisplayName></Owner><AccessControlList>'
    for identifier, name, permission in grants:
        xml += ('<Grant><Grantee xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:type="CanonicalUser">'
                f'<ID>{identifier}</ID><DisplayName>{name}</DisplayName></Grantee><Permission>{permission}</Permission></Grant>')
    return RawResponse(data=(xml + "</AccessControlList></AccessControlPolicy>").encode())


class RawConnection:
    """Replace only HTTP I/O; real S3 requests, signing and errors still execute."""
    def __init__(self, fake, host):
        self.fake = fake
        self.host = host

    def request(self, method, uri, body, headers):
        assert "Authorization" in headers
        parsed = urlsplit(uri)
        params = parse_qs(parsed.query, keep_blank_values=True)
        operation = "head" if method == "HEAD" else next(iter(params))
        assert method == ("HEAD" if operation == "head" else "GET")
        assert not body
        assert operation in {"versioning", "acl", "policy", "head", "location"}
        path = unquote(parsed.path).rstrip("/")
        # Redirects can switch virtual-host requests to path-style routing.
        bucket_path = "/lm-private-backup-sfx"
        if path == bucket_path or path.startswith(bucket_path + "/"):
            path = path[len(bucket_path):]
        target = BUCKET + path
        self.fake.raw_calls.append((operation, target))
        self.fake.routes.append((self.host, method, uri))
        self.response = self.fake.raw_hook(operation, target)
        if self.response is not None:
            return
        if operation == "versioning":
            value = "" if self.fake.versioning == "none" else f"<Status>{self.fake.versioning}</Status>"
            self.response = RawResponse(data=f'<VersioningConfiguration xmlns="{NAMESPACE}">{value}</VersioningConfiguration>'.encode())
        elif operation == "policy":
            self.response = raw_error(404, "NoSuchBucketPolicy") if self.fake.policy == "none" else RawResponse(data=self.fake.policy.encode())
        elif operation == "acl":
            grants = [("fixture-owner", "sfx", "FULL_CONTROL")]
            if target in self.fake.public:
                grants.append(("foreign-id", "foreign", "READ"))
            self.response = raw_acl(grants)
        elif operation == "head":
            payload = self.fake.objects[target]
            self.response = RawResponse(headers={"content-length": str(len(payload)), "etag": f'"{hashlib.md5(payload).hexdigest()}"'})
        elif operation == "location":
            self.response = RawResponse(data=f'<LocationConstraint xmlns="{NAMESPACE}">fixture-1</LocationConstraint>'.encode())

    def getresponse(self):
        return self.response


class FakeS3:
    def __init__(self, home):
        self.home = home
        self.objects = {}
        self.calls = []
        self.raw_calls = []
        self.routes = []
        self.raw_hook = lambda operation, target: None
        self.hook = lambda operation, args: None
        self.public = set()
        self.versioning = "none"
        self.policy = "none"
        self.uploads = []

    def __call__(self, command, **kwargs):
        config = str(self.home / ".config/living-memory/s3cmd-backup.conf")
        if command[0] == "/usr/bin/python3":
            assert command[1] == "-c"
            assert command[3:5] == [str(SCRIPT.resolve()), config]
            operation, args = "info", [command[5]]
            assert kwargs["timeout"] == 120
        else:
            assert command[:3] == ["/usr/bin/s3cmd", "-c", config]
            args = [arg for arg in command[3:] if not arg.startswith("--")]
            operation, *args = args
            assert operation != "info", "human-display metadata is forbidden"
            assert "--max-retries=0" in command
        # Every remote read/mutation belongs to the actual command's lock.
        with (self.home / ".local/state/living-memory-backup/backup.lock").open("a") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert 0 < kwargs["timeout"] <= 3600
        self.calls.append((operation, tuple(args)))
        override = self.hook(operation, args)
        if override is not None:
            return override
        output = ""
        if operation == "info":
            stdout, stderr = io.StringIO(), io.StringIO()
            disabled, timeout = logging.root.manager.disable, socket.getdefaulttimeout()
            try:
                with pytest.MonkeyPatch.context() as patch, redirect_stdout(stdout), redirect_stderr(stderr):
                    # A new metadata subprocess has fresh S3 singleton/maps.
                    patch.setattr(Config, "_instance", None)
                    patch.setattr(S3Request, "region_map", {})
                    patch.setattr(S3Request, "redir_map", {})
                    patch.setattr(ConnMan, "get", lambda host: SimpleNamespace(path=None, c=RawConnection(self, host)))
                    patch.setattr(ConnMan, "put", lambda conn: None)
                    patch.setattr(ConnMan, "close", lambda conn: None)
                    patch.setattr(sys, "argv", ["-c", *command[3:]])
                    exec(command[2], {"__name__": "__main__"})
                    assert Config().max_retries == 0
            finally:
                logging.disable(disabled)
                socket.setdefaulttimeout(timeout)
            assert stderr.getvalue() == ""
            output = stdout.getvalue()
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
    config.write_text("[default]\naccess_key = fixture\nsecret_key = fixture\n"
                      "host_base = storage.invalid\nhost_bucket = %(bucket)s.storage.invalid\n"
                      "bucket_location = fixture-1\n")
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
    # Bucket: versioning/ACL/policy. Content comparison: two HEADs. Every
    # download verification still gets its own ACL/policy; no display reads.
    assert Counter(op for op, _ in backup.raw_calls) == {
        "versioning": 1, "acl": 6, "policy": 6, "head": 2,
    }
    safety = next(target for op, target in backup.raw_calls if "/.previous-" in target)
    assert Counter(backup.raw_calls) == {
        ("versioning", BUCKET): 1, ("acl", BUCKET): 1, ("policy", BUCKET): 6,
        ("acl", FINAL): 2, ("head", FINAL): 1,
        ("acl", safety): 2, ("head", safety): 1,
        ("acl", STAGING): 1,
    }


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
        if failed and operation == "cp" and args[-1] == FINAL:
            backup.public.discard(FINAL)
        if not failed and args[0] == FINAL and backup.objects[FINAL] != backup.old:
            if failure == "acl" and operation == "info":
                failed = True
                backup.public.add(FINAL)
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
    assert Counter(op for op, _ in backup.raw_calls) == {"versioning": 1, "acl": 3, "policy": 3}


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


@pytest.mark.parametrize("response", [
    raw_error(403, "AccessDenied"), raw_error(500, "InternalError"),
    raw_error(503, "ServiceUnavailable"), raw_error(404, "NoSuchBucket"),
    RawResponse(data=b""), RawResponse(data=b"<VersioningConfiguration>"),
    RawResponse(data=b"<Error/>"),
    RawResponse(data=b'<VersioningConfiguration xmlns="urn:wrong"/>'),
    RawResponse(data=b"<VersioningConfiguration>none</VersioningConfiguration>"),
    RawResponse(data=b"<VersioningConfiguration><Status/></VersioningConfiguration>"),
    RawResponse(data=b"<VersioningConfiguration><Status>Enabled</Status></VersioningConfiguration>"),
    RawResponse(data=b"<VersioningConfiguration><Status>Suspended</Status></VersioningConfiguration>"),
    RawResponse(data=b"<VersioningConfiguration><Unknown/></VersioningConfiguration>"),
    RawResponse(data=b"<VersioningConfiguration><MfaDelete>Disabled</MfaDelete></VersioningConfiguration>"),
    RawResponse(204, b"<VersioningConfiguration/>"),
], ids=["403", "500", "503", "404", "empty", "broken-xml", "wrong-root", "wrong-namespace",
        "text", "empty-status", "enabled", "suspended", "unknown-child", "mfa-only", "204"])
def test_raw_versioning_cannot_authorize_mutation(backup, response, capsys):
    backup.raw_hook = lambda op, target: response if op == "versioning" else None
    assert backup.invoke() == 1
    assert backup.objects == {FINAL: backup.old}
    assert not any(op in {"cp", "put", "del", "abortmp"} for op, _ in backup.calls)
    assert backup.raw_calls == [("versioning", BUCKET)]  # No retry on 5xx.
    status = backup.status()
    assert "versioning" in status["last_error"]["message"]
    assert "FAKE SECRET" not in json.dumps(status) + str(capsys.readouterr())


@pytest.mark.parametrize("response", [
    raw_error(403, "AccessDenied"), raw_error(500, "InternalError"),
    raw_error(503, "ServiceUnavailable"), raw_error(501, "NotImplemented"),
    raw_error(404, "NoSuchBucket"), RawResponse(404),
    RawResponse(404, b"<Error>"), raw_error(403, "NoSuchBucketPolicy"),
    RawResponse(data=b"none"), RawResponse(data=b""),
    RawResponse(data=b'{"Statement": []}'), RawResponse(data=b"{broken"),
], ids=["403", "500", "503", "501", "wrong-404", "bare-404", "broken-404",
        "wrong-status", "display-none", "empty", "existing-policy", "malformed-policy"])
@pytest.mark.parametrize("target", [BUCKET, FINAL])
def test_raw_policy_cannot_authorize_mutation(backup, response, target, capsys):
    # Policy is bucket-scoped even when the consumer checks an object's privacy.
    failed = False

    def raw_hook(op, uri):
        nonlocal failed
        if op == "policy" and (target == BUCKET or ("acl", FINAL) in backup.raw_calls):
            assert uri == BUCKET
            failed = True
            return response

    backup.raw_hook = raw_hook
    assert backup.invoke() == 1
    assert failed
    assert backup.objects == {FINAL: backup.old}
    assert not any(op in {"cp", "put", "del", "abortmp"} for op, _ in backup.calls)
    assert sum(op == "policy" for op, _ in backup.raw_calls) == (1 if target == BUCKET else 2)
    assert "policy" in backup.status()["last_error"]["message"]
    assert "FAKE SECRET" not in json.dumps(backup.status()) + str(capsys.readouterr())


@pytest.mark.parametrize("xml", [
    b"<VersioningConfiguration/>",
    f'<VersioningConfiguration xmlns="{NAMESPACE}">\n </VersioningConfiguration>'.encode(),
])
def test_raw_confirmed_absence_allows_backup(backup, xml):
    backup.raw_hook = lambda op, target: RawResponse(data=xml) if op == "versioning" else None
    assert backup.invoke() == 0
    backup.fresh_bytes()


@pytest.mark.parametrize("operation", ["versioning", "acl", "policy", "head"])
def test_raw_transport_failure_is_not_absence(backup, operation, capsys):
    failed = 0

    def unavailable(op, target):
        nonlocal failed
        if op == operation:
            failed += 1
            raise socket.timeout("FAKE SECRET")

    backup.raw_hook = unavailable
    assert backup.invoke() == 1
    assert failed == 1
    assert backup.objects == {FINAL: backup.old}
    assert not any(op in {"cp", "put", "del", "abortmp"} for op, _ in backup.calls)
    assert operation in backup.status()["last_error"]["message"]
    assert "FAKE SECRET" not in json.dumps(backup.status()) + str(capsys.readouterr())


@pytest.mark.parametrize("target", [BUCKET, FINAL])
@pytest.mark.parametrize("grants", [
    [], [("fixture-owner", "sfx", "READ")],
    [("foreign-id", "foreign", "FULL_CONTROL")],
    [("foreign-id", "sfx", "FULL_CONTROL")],  # DisplayName cannot disguise a foreign ID.
    [("fixture-owner", "sfx", "FULL_CONTROL"), ("foreign-id", "foreign", "READ")],
])
def test_raw_acl_must_be_exactly_owner_full_control(backup, target, grants):
    backup.raw_hook = lambda op, uri: raw_acl(grants) if op == "acl" and uri == target else None
    assert backup.invoke() == 1
    assert backup.objects == {FINAL: backup.old}
    assert not any(op in {"cp", "put", "del", "abortmp"} for op, _ in backup.calls)


@pytest.mark.parametrize("failure", ["safety-download", "staging", "final"])
def test_policy_is_rechecked_at_each_verification(backup, failure):
    failed = False

    def policy_changed(op, target):
        nonlocal failed
        if op == "policy" and not failed:
            # The preceding ACL identifies the current verification; the
            # bucket policy itself is always read from the bucket root.
            acl_target = backup.raw_calls[-2][1]
            if ((failure == "safety-download" and "/.previous-" in acl_target
                 and backup.raw_calls.count(("acl", acl_target)) == 2)
                    or (failure == "staging" and acl_target == STAGING)
                    or (failure == "final" and acl_target == FINAL and backup.objects[FINAL] != backup.old)):
                failed = True
                return raw_error(403, "AccessDenied")

    backup.raw_hook = policy_changed
    assert backup.invoke() == 1
    assert failed
    assert backup.objects == {FINAL: backup.old}
    if failure == "safety-download":
        assert not any(op == "put" for op, _ in backup.calls)
    if failure != "final":
        assert not any(op == "cp" and args[-1] == FINAL for op, args in backup.calls)
    else:
        assert backup.status()["rollback"] == "verified"


@pytest.mark.parametrize("etag", ['"opaque-multipart-42"', '"' + '0' * 32 + '"'])
def test_original_attrs_md5_has_priority_over_etag(backup, etag):
    def multipart_headers(op, target):
        if op == "head":
            payload = backup.objects[target]
            return RawResponse(headers={
                "content-length": str(len(payload)), "etag": etag,
                "x-amz-meta-s3cmd-attrs": "uid:1000/md5:" + hashlib.md5(payload).hexdigest().upper(),
            })

    backup.raw_hook = multipart_headers
    assert backup.invoke() == 0
    backup.fresh_bytes()


@pytest.mark.parametrize("headers", [
    {"content-length": "42", "etag": '"' + 'a' * 32 + '-2"'},
    {"content-length": "42"},
    {"content-length": "42", "etag": '"' + 'a' * 32 + '"', "x-amz-meta-s3cmd-attrs": "md5:invalid"},
    {"content-length": "42", "x-amz-meta-s3cmd-attrs": "malformed"},
    {"content-length": "-1", "etag": '"' + 'a' * 32 + '"'},
    {"content-length": "1.0", "etag": '"' + 'a' * 32 + '"'},
    {"etag": '"' + 'a' * 32 + '"'},
], ids=["opaque-multipart", "missing-md5", "invalid-md5", "malformed-attrs", "negative-length", "fractional-length", "missing-length"])
def test_raw_content_metadata_fails_closed(backup, headers):
    backup.raw_hook = lambda op, target: RawResponse(headers=headers) if op == "head" else None
    assert backup.invoke() == 1
    assert backup.objects == {FINAL: backup.old}
    assert not any(op in {"cp", "put", "del", "abortmp"} for op, _ in backup.calls)


@pytest.mark.parametrize("incorrect", ["md5", "length"])
def test_metadata_must_match_actual_previous_content(backup, incorrect):
    def wrong_headers(op, target):
        if op == "head":
            payload = backup.objects[target]
            return RawResponse(headers={
                "content-length": "0" if incorrect == "length" else str(len(payload)),
                "etag": '"opaque-multipart-2"',
                "x-amz-meta-s3cmd-attrs": "md5:" + ("0" * 32 if incorrect == "md5" else hashlib.md5(payload).hexdigest()),
            })

    backup.raw_hook = wrong_headers
    assert backup.invoke() == 1
    assert backup.objects == {FINAL: backup.old}
    assert not any(op == "put" or (op == "cp" and args[-1] == FINAL) for op, args in backup.calls)


def test_metadata_keeps_transport_region_discovery(backup):
    config = backup.home / ".config/living-memory/s3cmd-backup.conf"
    config.write_text(config.read_text().replace("fixture-1", "US"))
    assert backup.invoke() == 0
    # Location is allowed only when the unchanged signing transport needs it.
    assert sum(op == "location" for op, _ in backup.raw_calls) == 6
    assert set(op for op, _ in backup.raw_calls) == {"location", "versioning", "acl", "policy", "head"}


def test_metadata_keeps_transport_redirects(backup):
    redirected = False

    def redirect(op, target):
        nonlocal redirected
        if op == "versioning" and not redirected:
            redirected = True
            return RawResponse(301, b"<Error><Code>PermanentRedirect</Code><Bucket>lm-private-backup-sfx</Bucket><Endpoint>redirect.invalid</Endpoint></Error>",
                               {"x-amz-bucket-region": "fixture-2"})

    backup.raw_hook = redirect
    assert backup.invoke() == 0
    assert backup.routes[0][0] == "lm-private-backup-sfx.storage.invalid"
    assert backup.routes[1][0] == "redirect.invalid"
    assert backup.raw_calls[:2] == [("versioning", BUCKET)] * 2


@pytest.mark.parametrize("failure", ["deadline", "spawn", "exit", "invalid-json"])
def test_metadata_subprocess_failure_stops_before_mutation(backup, failure, capsys):
    def unavailable(op, args):
        if op == "info":
            if failure == "deadline":
                raise subprocess.TimeoutExpired("FAKE SECRET", 120)
            if failure == "spawn":
                raise OSError("FAKE SECRET")
            return subprocess.CompletedProcess([], 1 if failure == "exit" else 0,
                                               "FAKE SECRET", "FAKE SECRET")

    backup.hook = unavailable
    assert backup.invoke() == 1
    assert backup.calls == [("info", (BUCKET,))]
    assert backup.objects == {FINAL: backup.old}
    assert "FAKE SECRET" not in json.dumps(backup.status()) + str(capsys.readouterr())
