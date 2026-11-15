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
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
from S3.Config import Config
from S3.ConnMan import ConnMan
from S3.S3 import S3, S3Request


SCRIPT = Path(__file__).with_name("backup_living_memory_s3.py")
BUCKET = "s3://lm-private-backup-sfx"
FINAL = BUCKET + "/global.sqlite3.gz"
STAGING = BUCKET + "/.incoming.sqlite3.gz"
NAMESPACE = "http://s3.amazonaws.com/doc/2006-03-01/"
REAL_SUBPROCESS_RUN = subprocess.run


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
        self.cli = lambda operation, args: False
        self.connection = lambda host: SimpleNamespace(path=None, c=RawConnection(self, host))

    def __call__(self, command, **kwargs):
        config = str(self.home / ".config/living-memory/s3cmd-backup.conf")
        assert command[:2] == ["/usr/bin/python3", "-c"]
        if "['metadata']" in command[2]:
            assert command[1] == "-c"
            assert command[3:5] == [str(SCRIPT.resolve()), config]
            operation, args = "info", [command[5]]
            assert kwargs["timeout"] == 120
        else:
            assert "['transport']" in command[2]
            assert command[3] == str(SCRIPT.resolve())
            assert kwargs["pass_fds"] == (int(command[4]),)
            self.diagnostic_fd = int(command[4])
            assert command[5:7] == ["-c", config]
            args = [arg for arg in command[7:] if not arg.startswith("--")]
            operation, *args = args
            assert operation != "info", "human-display metadata is forbidden"
            assert "--max-retries=0" in command
        # Every remote read/mutation belongs to the actual command's lock.
        with (self.home / ".local/state/living-memory-backup/backup.lock").open("a") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected_timeout = 3600 if operation in {"get", "put", "cp"} else 120
        if operation in {"put", "cp", "del", "abortmp"}:
            expected_timeout += 3279 + 7 * 30
        assert kwargs["timeout"] == expected_timeout
        self.calls.append((operation, tuple(args)))
        override = self.hook(operation, args)
        if override is not None:
            return override
        if self.cli(operation, args):
            return self.installed(command)
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

    def installed(self, command):
        """Run the actual child entry/CLI in process with only HTTP I/O faked."""
        stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        stderr = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        disabled, timeout = logging.root.manager.disable, socket.getdefaulttimeout()
        try:
            with pytest.MonkeyPatch.context() as patch, redirect_stdout(stdout), redirect_stderr(stderr):
                patch.setattr(Config, "_instance", None)
                patch.setattr(S3Request, "region_map", {})
                patch.setattr(S3Request, "redir_map", {})
                patch.setattr(ConnMan, "get", self.connection)
                patch.setattr(ConnMan, "put", lambda conn: None)
                patch.setattr(ConnMan, "close", lambda conn: None)
                for name in ("send_request", "send_file", "send_file_multipart", "recv_file"):
                    patch.setattr(S3, name, getattr(S3, name))
                patch.setattr(sys, "argv", ["-c", *command[3:]])
                with pytest.raises(SystemExit) as stopped:
                    exec(command[2], {"__name__": "__main__"})
                assert Config().max_retries == 0
                sys.stdout.flush()
            assert stderr.buffer.getvalue() == b""
            return subprocess.CompletedProcess(command, stopped.value.code,
                                               stdout.buffer.getvalue().decode(), "")
        finally:
            logging.disable(disabled)
            socket.setdefaulttimeout(timeout)


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
    fake.sleeps = []
    monkeypatch.setattr(time, "sleep", fake.sleeps.append)
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


@pytest.mark.parametrize("operation", ["info", "ls", "multipart", "get"])
@pytest.mark.parametrize("failures", [1, 7, 8])
def test_read_retry_budget_and_clean_downloads(backup, operation, failures):
    attempts = 0

    def transient(verb, args):
        nonlocal attempts
        if verb != operation or (verb == "get" and args[0] != STAGING):
            return
        # Only the first logical read fails; later cleanup reads succeed.
        if attempts > failures:
            return
        attempts += 1
        if verb == "get":
            assert not Path(args[1]).exists(), "retry must discard partial download"
        if attempts <= failures:
            if verb == "get":
                Path(args[1]).write_bytes(b"partial download")
            raise subprocess.TimeoutExpired("FAKE SECRET", 120)

    backup.hook = transient
    assert backup.invoke() == (1 if failures == 8 else 0)
    assert attempts == min(failures + 1, 8)
    assert backup.sleeps == [3, 9, 27, 81, 243, 729, 2187][:failures]
    if failures == 8:
        assert backup.objects == {FINAL: backup.old}
        assert backup.status()["last_error"]["transport"]["chain"] == [{"class": "TimeoutExpired"}]
    else:
        backup.fresh_bytes()
        assert backup.status()["last_error"] is None
        assert backup.status()["transaction"] is None


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 403, 404, 501])
def test_metadata_http_retry_classification(backup, status):
    attempts = 0

    def unavailable(operation, target):
        nonlocal attempts
        if operation == "versioning":
            attempts += 1
            return raw_error(status, "FixtureFailure")

    backup.raw_hook = unavailable
    assert backup.invoke() == 1
    retryable = status in {408, 429, 500, 502, 503, 504}
    assert attempts == (8 if retryable else 1)
    assert backup.sleeps == ([3, 9, 27, 81, 243, 729, 2187] if retryable else [])
    assert backup.objects == {FINAL: backup.old}
    assert not any(op in {"cp", "put", "del", "abortmp"} for op, _ in backup.calls)


def test_read_retry_stops_when_failure_becomes_permanent(backup):
    attempts = 0

    def failure(operation, args):
        nonlocal attempts
        if operation == "ls":
            attempts += 1
            if attempts == 1:
                raise subprocess.TimeoutExpired("FAKE SECRET", 120)
            os.write(backup.diagnostic_fd, b'[{"class":"S3Error","status":403}]')
            return subprocess.CompletedProcess([], 1, "", "FAKE SECRET")

    backup.hook = failure
    assert backup.invoke() == 1
    assert attempts == 2
    assert backup.sleeps == [3]
    assert backup.status()["last_error"]["transport"]["chain"] == [{"class": "S3Error", "status": 403}]
    assert backup.objects == {FINAL: backup.old}


@pytest.mark.parametrize("operation", ["put", "cp", "del", "abortmp"])
def test_mutation_timeout_has_no_automatic_retry(backup, operation):
    attempts = 0

    def timeout(verb, args):
        nonlocal attempts
        if operation == "abortmp" and verb == "get" and args[0] == FINAL:
            backup.uploads.append((STAGING, "owned-upload"))
        if verb == operation:
            attempts += 1
            raise subprocess.TimeoutExpired("FAKE SECRET", 120)

    backup.hook = timeout
    assert backup.invoke() == 3
    assert attempts == 1
    assert backup.sleeps == []
    assert backup.status()["transaction"]["pending"]["operation"] == operation


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
        if (failure == "timeout" and operation == "get" and args[0] == FINAL
                and backup.objects[FINAL] != backup.old):
            failed = True
            raise subprocess.TimeoutExpired("FAKE SECRET", 120)
        if not failed and args[0] == FINAL and backup.objects[FINAL] != backup.old:
            if failure == "acl" and operation == "info":
                failed = True
                backup.public.add(FINAL)
            if operation == "get":
                failed = True
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
    assert all(op in {"info", "ls", "multipart", "get"} for op, _ in backup.calls[len(calls):])
    assert backup.status()["recovery_error"]


@pytest.mark.parametrize("operation", ["safety", "upload", "replace"])
@pytest.mark.parametrize("applied", [False, True])
def test_unknown_mutations_are_reconciled_before_continuing(backup, operation, applied):
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
    safety = backup.status()["transaction"]["safety"]
    assert backup.status()["result"] == "unknown"
    if operation != "safety" or applied:
        assert backup.objects[safety] == backup.old
    backup.hook = lambda *args: None
    backup.calls.clear()
    assert backup.invoke() == 0
    backup.fresh_bytes()
    assert set(backup.objects) == {FINAL}
    assert backup.status()["transaction"] is None
    mutations = [op for op, _ in backup.calls if op in {"put", "cp", "del", "abortmp"}]
    assert backup.calls[0][0] == "info"
    assert "get" in [op for op, _ in backup.calls[:next(
        i for i, (op, _) in enumerate(backup.calls) if op in mutations)]]
    if operation == "replace" or (operation == "upload" and applied):
        assert "put" not in mutations
    else:
        assert mutations.count("put") == 1
    if operation == "replace" and applied:
        assert "cp" not in mutations


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
    backup.hook = lambda *args: None
    backup.calls.clear()
    assert backup.invoke() == 0
    deleted = STAGING if target == "staging" else status["transaction"]["safety"]
    assert not any(op == "del" and args[0] == deleted for op, args in backup.calls)
    assert not any(op in {"put", "cp", "get"} for op, _ in backup.calls)
    assert set(backup.objects) == {FINAL}
    assert backup.status()["last_success"] == status["last_success"]


def interrupted_upload(backup, *, complete=False):
    def timeout(operation, args):
        if operation == "put":
            if complete:
                backup.objects[STAGING] = Path(args[0]).read_bytes()
            else:
                backup.uploads.append((STAGING, "interrupted-upload"))
            raise subprocess.TimeoutExpired("fixture upload", 3600)

    backup.hook = timeout
    assert backup.invoke() == 3
    backup.hook = lambda *args: None
    backup.calls.clear()


def test_scheduled_multipart_recovery_takes_current_snapshot(backup, tmp_path):
    interrupted_upload(backup)
    old_expected = backup.status()["transaction"]["expected"]
    with sqlite3.connect(backup.database) as connection:
        connection.execute("CREATE TABLE new_fact (value TEXT)")
        connection.execute("INSERT INTO new_fact VALUES ('after timeout')")

    def check_preservation(operation, args):
        if operation in {"abortmp", "put"}:
            assert backup.objects[FINAL] == backup.old
        if operation == "put":
            assert backup.uploads == []
        if operation == "del" and args[0] != STAGING and backup.objects[FINAL] != backup.old:
            assert backup.status()["result"] == "verified"

    backup.hook = check_preservation
    assert backup.invoke() == 0
    assert backup.uploads == []
    assert set(backup.objects) == {FINAL}
    assert backup.status()["sha256"] != old_expected["sha256"]
    restored = tmp_path / "recovered.sqlite3"
    restored.write_bytes(gzip.decompress(backup.objects[FINAL]))
    with sqlite3.connect(restored) as connection:
        assert connection.execute("SELECT value FROM new_fact").fetchall() == [("after timeout",)]
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    assert sum(op == "put" for op, _ in backup.calls) == 1
    assert [(op, args) for op, args in backup.calls if op == "abortmp"] == [
        ("abortmp", (STAGING, "interrupted-upload"))]


@pytest.mark.parametrize("failure", ["info", "ls", "multipart", "final_read", "safety_read",
                                    "public", "versioning", "foreign_object", "foreign_upload",
                                    "bad_final", "bad_safety", "bad_staging", "missing_safety"])
def test_recovery_requires_positive_remote_evidence(backup, failure):
    previous = seed_success(backup)
    interrupted_upload(backup)
    status = backup.status()
    safety = status["transaction"]["safety"]
    if failure == "public":
        backup.public.add(safety)
    elif failure == "versioning":
        backup.versioning = "enabled"
    elif failure == "foreign_object":
        backup.objects[BUCKET + "/unrelated"] = b"keep"
    elif failure == "foreign_upload":
        backup.uploads.append((BUCKET + "/unrelated", "keep"))
    elif failure in {"bad_final", "bad_safety", "bad_staging"}:
        backup.objects[{"bad_final": FINAL, "bad_safety": safety, "bad_staging": STAGING}[failure]] = b"corrupt"
    elif failure == "missing_safety":
        del backup.objects[safety]
    else:
        def unavailable(operation, args):
            if (operation == failure or (operation == "get" and args[0] ==
                    {"final_read": FINAL, "safety_read": safety}.get(failure))):
                raise subprocess.TimeoutExpired("fixture recovery read", 120)
        backup.hook = unavailable
    objects, uploads = dict(backup.objects), list(backup.uploads)
    assert backup.invoke() == 3
    assert backup.objects == objects
    assert backup.uploads == uploads
    assert backup.status()["transaction"] == status["transaction"]
    assert backup.status()["last_success"] == previous["last_success"]
    assert backup.status()["recovery_error"]
    assert not any(op in {"cp", "put", "del", "abortmp"} for op, _ in backup.calls)


def test_transient_recovery_read_failure_recovers_on_next_schedule(backup):
    interrupted_upload(backup)

    def unavailable(operation, args):
        if operation == "ls":
            raise subprocess.TimeoutExpired("fixture listing", 120)

    backup.hook = unavailable
    assert backup.invoke() == 3
    assert backup.status()["recovery_error"]
    backup.hook = lambda *args: None
    assert backup.invoke() == 0
    backup.fresh_bytes()
    assert backup.status()["recovery_error"] is None


@pytest.mark.parametrize("operation", ["abortmp", "del"])
@pytest.mark.parametrize("applied", [False, True])
def test_recovery_cleanup_survives_another_timeout(backup, operation, applied):
    interrupted_upload(backup)
    safety = backup.status()["transaction"]["safety"]

    def lost_response(verb, args):
        if verb == operation:
            if applied:
                if verb == "abortmp":
                    backup.uploads.remove(tuple(args))
                else:
                    backup.objects.pop(args[0])
            raise subprocess.TimeoutExpired("fixture recovery cleanup", 120)

    backup.hook = lost_response
    assert backup.invoke() == 3
    assert backup.objects[FINAL] == backup.old
    assert backup.status()["transaction"]["pending"]["operation"] == operation
    backup.hook = lambda *args: None
    backup.calls.clear()
    assert backup.invoke() == 0
    backup.fresh_bytes()
    assert set(backup.objects) == {FINAL}
    assert backup.uploads == []
    if applied:
        assert not any(op == "abortmp" for op, _ in backup.calls)
        if operation == "del":
            assert not any(op == "del" and args[0] == safety for op, args in backup.calls)


@pytest.mark.parametrize("complete", [False, True])
def test_first_backup_upload_recovery(backup, complete):
    backup.objects.clear()
    interrupted_upload(backup, complete=complete)
    assert backup.invoke() == 0
    backup.fresh_bytes()
    assert set(backup.objects) == {FINAL}
    assert backup.uploads == []
    assert sum(op == "put" for op, _ in backup.calls) == (0 if complete else 1)


def test_recovery_interrupted_after_remote_replace(backup):
    interrupted_upload(backup, complete=True)

    def crash(operation, args):
        if operation == "cp" and args[-1] == FINAL:
            backup.objects[FINAL] = backup.objects[STAGING]
            raise KeyboardInterrupt("fixture recovery crash")

    backup.hook = crash
    with pytest.raises(KeyboardInterrupt):
        backup.invoke()
    assert backup.status()["transaction"]["pending"] == {"operation": "cp", "target": FINAL}
    backup.hook = lambda *args: None
    backup.calls.clear()
    assert backup.invoke() == 0
    backup.fresh_bytes()
    assert not any(op in {"put", "cp"} for op, _ in backup.calls)


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


class TransferConnection:
    """Allow real CLI HEAD/setup, then fail in the installed transfer method."""
    def __init__(self, failure, calls):
        self.failure, self.calls = failure, calls

    def request(self, method, uri, body, headers):
        assert "Authorization" in headers
        self.calls.append(method)
        if method == "HEAD":
            self.response = RawResponse(headers={
                "content-length": "1", "etag": '"0123456789abcdef0123456789abcdef"',
                "last-modified": "Thu, 01 Jan 2026 00:00:00 GMT"})
        elif method == "POST":
            assert "uploads" in uri
            self.response = RawResponse(data=b"<InitiateMultipartUploadResult><UploadId>FAKE-SECRET-ID</UploadId></InitiateMultipartUploadResult>")
        else:
            if isinstance(self.failure, Exception):
                raise self.failure
            self.response = self.failure

    def putrequest(self, method, uri):
        self.calls.append(method)
        if isinstance(self.failure, Exception):
            raise self.failure
        self.response = self.failure

    def putheader(self, *args):
        pass

    def endheaders(self):
        pass

    def wrapper_send_body(self, data):
        pass

    def getresponse(self):
        return self.response


@pytest.mark.parametrize("failure", ["connect", "body", "lost_response", "denied"])
@pytest.mark.parametrize("failures", [1, 7, 8])
def test_multipart_retries_keep_part_identity_and_bytes(backup, monkeypatch, failure, failures):
    from S3.MultiPart import MultiPartUpload
    from S3.S3Uri import S3Uri

    namespace = runpy.run_path(str(SCRIPT))
    monkeypatch.setattr(Config, "_instance", None)
    cfg = Config(str(backup.home / ".config/living-memory/s3cmd-backup.conf"))
    cfg.max_retries, cfg.progress_meter, cfg.use_http_expect = 0, False, False
    monkeypatch.setattr(S3Request, "region_map", {})
    monkeypatch.setattr(S3Request, "redir_map", {})
    monkeypatch.setattr(ConnMan, "conn_pool", {})
    part = b"second part bytes" * 8192
    prefix = b"first part differs" * 8192
    stream = io.BytesIO(prefix + part)
    stream.stream_name = "private fixture archive"
    upload = MultiPartUpload.__new__(MultiPartUpload)
    upload.s3, upload.file_stream = S3(cfg), stream
    upload.dst_uri, upload.upload_id = S3Uri(STAGING), "fixed-upload-id"
    upload.parts = {1: "previous-part-etag"}
    accepted, requests, connections = {}, [], []
    attempts = 0

    class Connection:
        def __init__(self, attempt):
            self.attempt, self.body, self.closed = attempt, bytearray(), False

        def putrequest(self, method, uri):
            params = parse_qs(urlsplit(uri).query)
            requests.append((method, params['uploadId'][0], params['partNumber'][0]))

        def putheader(self, *args):
            pass

        def endheaders(self):
            pass

        def wrapper_send_body(self, data):
            self.body.extend(data)
            if failure == "body" and self.attempt <= failures:
                raise ConnectionResetError(104, "FAKE SECRET")

        def getresponse(self):
            if failure == "denied":
                return raw_error(403, "AccessDenied")
            assert bytes(self.body) == part
            accepted[2] = bytes(self.body)
            if failure == "lost_response" and self.attempt <= failures:
                # The part was stored; losing its response must not append data.
                stream.seek(len(prefix))
                stream.write(b"!" * len(part))
                raise TimeoutError(110, "FAKE SECRET")
            return RawResponse(headers={"etag": '"' + hashlib.md5(part).hexdigest() + '"'})

        def close(self):
            self.closed = True

    def acquire(host, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        assert all(conn.closed for conn in connections), "retry must discard failed connections"
        if failure == "connect" and attempts <= failures:
            raise TimeoutError(110, "FAKE SECRET")
        conn = Connection(attempts)
        connections.append(conn)
        return SimpleNamespace(c=conn, path=None)

    monkeypatch.setattr(ConnMan, "get", acquire)
    monkeypatch.setattr(ConnMan, "put", lambda conn: None)
    monkeypatch.setattr(ConnMan, "close", lambda conn: conn.c.close())
    original_part = MultiPartUpload.upload_part
    with namespace["resilient_transport"](True):
        if failure == "denied" or failures == 8:
            with pytest.raises(Exception):
                upload.upload_part(2, len(prefix), len(part), {})
            assert upload.parts == {1: "previous-part-etag"}
        else:
            upload.upload_part(2, len(prefix), len(part), {})
            assert upload.parts == {1: "previous-part-etag", 2: hashlib.md5(part).hexdigest()}
            assert accepted == {2: part}
    assert MultiPartUpload.upload_part is original_part
    assert attempts == (1 if failure == "denied" else min(failures + 1, 8))
    assert backup.sleeps == ([] if failure == "denied" else [3, 9, 27, 81, 243, 729, 2187][:failures])
    assert all(item == ('PUT', 'fixed-upload-id', '2') for item in requests)
    assert all(part.startswith(bytes(conn.body)) for conn in connections)
    assert all(conn.closed for conn in connections)


def test_command_completes_after_lost_multipart_response(backup):
    with sqlite3.connect(backup.database) as connection:
        connection.execute("CREATE TABLE padding (value BLOB)")
        connection.execute("INSERT INTO padding VALUES (?)", (os.urandom(6 * 1024 * 1024),))
    with (backup.home / ".config/living-memory/s3cmd-backup.conf").open("a") as stream:
        stream.write("multipart_chunk_size_mb = 5\nuse_http_expect = False\n")
    parts, attempts = {}, Counter()
    posts = []

    class Connection:
        def request(self, method, uri, body, headers):
            self.part = None
            if method == "HEAD":
                self.response = RawResponse(headers={'content-length': '1',
                    'etag': '"0123456789abcdef0123456789abcdef"',
                    'last-modified': 'Thu, 01 Jan 2026 00:00:00 GMT'})
                return
            assert method == "POST"
            params = parse_qs(urlsplit(uri).query, keep_blank_values=True)
            if "uploads" in params:
                posts.append('initiate')
                self.response = RawResponse(data=b'<InitiateMultipartUploadResult><UploadId>fixture-upload</UploadId></InitiateMultipartUploadResult>')
            else:
                assert params['uploadId'] == ['fixture-upload']
                posts.append('complete')
                assert set(parts) == {1, 2}
                backup.objects[STAGING] = parts[1] + parts[2]
                self.response = RawResponse(data=b'<CompleteMultipartUploadResult><ETag>fixture-etag</ETag></CompleteMultipartUploadResult>')

        def putrequest(self, method, uri):
            assert method == 'PUT'
            params = parse_qs(urlsplit(uri).query)
            assert params['uploadId'] == ['fixture-upload']
            self.part = int(params['partNumber'][0])
            attempts[self.part] += 1
            self.body = bytearray()

        def putheader(self, *args):
            pass

        def endheaders(self):
            pass

        def wrapper_send_body(self, data):
            self.body.extend(data)

        def getresponse(self):
            if self.part is None:
                return self.response
            body = bytes(self.body)
            if self.part in parts:
                assert parts[self.part] == body
            parts[self.part] = body
            if self.part == 2 and attempts[2] < 3:
                raise TimeoutError(110, 'FAKE SECRET response lost after storing part')
            return RawResponse(headers={'etag': '"' + hashlib.md5(body).hexdigest() + '"'})

    backup.cli = lambda operation, args: operation == 'put'
    backup.connection = lambda host: SimpleNamespace(path=None, c=Connection())
    assert backup.invoke() == 0
    backup.fresh_bytes()
    assert backup.status()['result'] == 'verified'
    assert backup.status()['cleanup'] == 'complete'
    assert backup.status()['transaction'] is None
    assert set(backup.objects) == {FINAL}
    assert attempts == {1: 1, 2: 3}
    assert posts == ['initiate', 'complete']
    assert backup.sleeps == [3, 9]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("permanent", [False, True])
def test_connection_setup_retry_budget_before_requests(backup, monkeypatch, enabled, permanent):
    namespace = runpy.run_path(str(SCRIPT))
    attempts = 0
    monkeypatch.setattr(ConnMan, "conn_pool", {})

    def unavailable(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if permanent:
            raise ValueError("invalid connection configuration")
        raise TimeoutError(110, "FAKE SECRET")

    monkeypatch.setattr(ConnMan, "get", unavailable)
    with namespace["resilient_transport"](enabled):
        with pytest.raises((TimeoutError, ValueError)):
            ConnMan.get("storage.invalid")
    assert attempts == (8 if enabled and not permanent else 1)
    assert backup.sleeps == ([3, 9, 27, 81, 243, 729, 2187] if enabled and not permanent else [])


@pytest.mark.parametrize("operation,params", [
    ("OBJECT_POST", {"uploads": None}),
    ("OBJECT_POST", {"uploadId": "fixed-upload"}),
    ("OBJECT_DELETE", {"uploadId": "fixed-upload"}),
    ("OBJECT_PUT", {}),
])
def test_unknown_nonpart_request_is_not_replayed(backup, monkeypatch, operation, params):
    from S3.S3Uri import S3Uri
    namespace = runpy.run_path(str(SCRIPT))
    monkeypatch.setattr(Config, "_instance", None)
    cfg = Config(str(backup.home / ".config/living-memory/s3cmd-backup.conf"))
    cfg.max_retries = 0
    monkeypatch.setattr(S3Request, "region_map", {})
    monkeypatch.setattr(S3Request, "redir_map", {})
    monkeypatch.setattr(ConnMan, "conn_pool", {})
    sent = []

    def send(method, uri, body, headers):
        sent.append(method)
        raise TimeoutError(110, "FAKE SECRET after request")

    conn = SimpleNamespace(path=None, c=SimpleNamespace(request=send))
    monkeypatch.setattr(ConnMan, "get", lambda *a, **k: conn)
    monkeypatch.setattr(ConnMan, "close", lambda conn: None)
    client = S3(cfg)
    with namespace["resilient_transport"](True):
        with pytest.raises(Exception):
            client.send_request(client.create_request(operation, uri=S3Uri(STAGING), uri_params=params))
    assert len(sent) == 1
    assert backup.sleeps == []


@pytest.mark.parametrize("secure", [False, True])
@pytest.mark.parametrize("timeout", [None, 3, 300])
def test_connect_deadline_restores_transfer_timeout(backup, monkeypatch, secure, timeout):
    import http.client
    namespace = runpy.run_path(str(SCRIPT))
    cls = http.client.HTTPSConnection if secure else http.client.HTTPConnection
    observed = []

    class Socket:
        def fileno(self):
            return 1

        def settimeout(self, value):
            observed.append(('transfer', value))

    def connect(conn):
        observed.append(('connect', conn.timeout))
        conn.sock = Socket()

    monkeypatch.setattr(cls, "connect", connect)
    with namespace["resilient_transport"](True):
        conn = SimpleNamespace(timeout=timeout, sock=None)
        cls.connect(conn)
    assert observed == [('connect', 3 if timeout == 3 else 15), ('transfer', timeout)]
    assert conn.timeout == timeout
    assert cls.connect is connect


@pytest.mark.parametrize("operation", ["put", "multipart-put", "get", "ls"])
@pytest.mark.parametrize("failure,tail", [
    (TimeoutError(110, "FAKE SECRET https://signed.invalid/?token=secret"), {"class": "TimeoutError", "errno": 110}),
    (ConnectionResetError(104, "FAKE SECRET Authorization: credential"), {"class": "ConnectionResetError", "errno": 104}),
    (raw_error(403, "AccessDenied"), {"class": "S3Error", "status": 403}),
])
def test_installed_transport_cause_reaches_status(backup, operation, failure, tail, capsys):
    previous = seed_success(backup)
    verb = "put" if operation == "multipart-put" else operation
    calls = []
    backup.cli = lambda op, args: op == verb
    backup.connection = lambda host: SimpleNamespace(path=None, c=TransferConnection(failure, calls))
    config = backup.home / ".config/living-memory/s3cmd-backup.conf"
    with config.open("a") as stream:
        stream.write("use_http_expect = False\n")
    if operation == "multipart-put":
        # Incompressible disposable rows select the installed multipart path.
        with sqlite3.connect(backup.database) as connection:
            connection.execute("CREATE TABLE padding (value BLOB)")
            connection.execute("INSERT INTO padding VALUES (?)", (os.urandom(6 * 1024 * 1024),))
        with config.open("a") as stream:
            stream.write("multipart_chunk_size_mb = 5\n")
    assert backup.invoke() == (3 if verb == "put" else 1)
    status = backup.status()
    diagnostic = status["last_error"]["transport"]
    assert diagnostic["operation"] == verb
    assert diagnostic["chain"][-1] == tail
    if isinstance(failure, Exception):
        root = {"put": "S3UploadError", "get": "S3DownloadError", "ls": "S3RequestError"}[verb]
        direct = isinstance(failure, ConnectionResetError) and verb != "put"
        assert diagnostic["chain"] == ([tail] if direct else [{"class": root}, tail])
        assert diagnostic["exit_code"] == (74 if direct else (75 if verb == "ls" else 2))
    attempts = 8 if operation in {"get", "ls", "multipart-put"} and isinstance(failure, Exception) else 1
    assert calls.count({"put": "PUT", "get": "GET", "ls": "GET"}[verb]) == attempts
    assert calls.count("POST") == (1 if operation == "multipart-put" else 0)
    assert backup.objects[FINAL] == backup.old
    assert status["last_success"] == previous["last_success"]
    if verb == "put":
        assert status["transaction"]["pending"]["operation"] == "put"
        assert backup.objects[status["transaction"]["safety"]] == backup.old
        backup.calls.clear()
        assert backup.invoke() == 3
        assert sum(op == "put" for op, _ in backup.calls) == 1
        assert backup.objects[FINAL] == backup.old
        assert backup.objects[backup.status()["transaction"]["safety"]] == backup.old
        assert backup.status()["last_error"]["transport"] == diagnostic
        assert backup.status()["last_error"]["message"] == status["last_error"]["message"]
    report = json.dumps(status["last_error"])
    output = str(capsys.readouterr())
    for secret in ("FAKE SECRET", "FAKE-SECRET-ID", "signed.invalid", "Authorization", "credential", "s3://", str(config)):
        assert secret not in report + output


@pytest.mark.parametrize("diagnostic", ["", "FAKE SECRET", '{"message":"FAKE SECRET"}',
                                        '[{"class":"TimeoutError","errno":"FAKE SECRET"}]'])
def test_unclassified_exit_never_invents_a_cause(backup, diagnostic, capsys):
    def failed(operation, args):
        if operation == "put":
            # The parent accepts neither raw CLI output nor an invalid envelope.
            os.write(backup.diagnostic_fd, diagnostic.encode())
            return subprocess.CompletedProcess([], 2, "FAKE SECRET", "FAKE SECRET")

    backup.hook = failed
    assert backup.invoke() == 3
    error = backup.status()["last_error"]
    assert error["transport"] == {"operation": "put", "exit_code": 2, "chain": []}
    assert "cause unavailable" in error["message"]
    assert "Timeout" not in error["message"]
    assert "FAKE SECRET" not in json.dumps(error) + str(capsys.readouterr())
    backup.calls.clear()
    assert backup.invoke() == 3
    assert sum(op == "put" for op, _ in backup.calls) == 1
    assert backup.objects[FINAL] == backup.old
    assert backup.status()["last_error"]["transport"] == error["transport"]


def test_exception_chain_is_bounded_and_uses_only_safe_fields(backup):
    namespace = runpy.run_path(str(SCRIPT))
    errors = [RuntimeError("FAKE SECRET") for _ in range(12)]
    for parent, child in zip(errors, errors[1:]):
        parent.__context__ = child
    assert len(namespace["exception_chain"](errors[0])) == 8
    error = OSError(110, "FAKE SECRET", "FAKE SECRET file")
    error.status = True
    error.__context__ = error
    assert namespace["exception_chain"](error) == [{"class": "TimeoutError", "errno": 110}]
    error.__cause__ = ConnectionError("FAKE SECRET")
    assert namespace["exception_chain"](error)[-1] == {"class": "ConnectionError"}


def test_diagnostic_descriptor_crosses_real_subprocess(backup, monkeypatch):
    namespace = runpy.run_path(str(SCRIPT))

    def isolated_child(command, **kwargs):
        command = list(command)
        command[2] = """
import socket
from S3.S3 import S3
from S3.ConnMan import ConnMan
def forbidden(*args, **kwargs):
    raise AssertionError('network forbidden')
socket.socket = forbidden
def unavailable(host):
    raise TimeoutError(110, 'FAKE SECRET https://signed.invalid Authorization: secret')
ConnMan.get = unavailable
""" + command[2]
        result = REAL_SUBPROCESS_RUN(command, **kwargs, cwd=backup.home,
                                     env={"HOME": str(backup.home), "LC_ALL": "C.UTF-8"})
        assert "FAKE SECRET" not in result.stdout + result.stderr
        assert result.stderr == ""
        return result

    monkeypatch.setattr(subprocess, "run", isolated_child)
    with pytest.raises(namespace["S3Failure"]) as stopped:
        namespace["s3"]("ls", BUCKET + "/")
    assert stopped.value.transport == {
        "operation": "ls", "exit_code": 75,
        "chain": [{"class": "S3RequestError"}, {"class": "TimeoutError", "errno": 110}],
    }


@pytest.mark.parametrize("failure,chain", [
    (subprocess.TimeoutExpired("FAKE SECRET command", 3600, output="FAKE SECRET", stderr="FAKE SECRET"),
     [{"class": "TimeoutExpired"}]),
    (OSError(2, "FAKE SECRET executable", "FAKE SECRET config"), [{"class": "FileNotFoundError", "errno": 2}]),
])
def test_subprocess_failure_has_its_own_typed_cause(backup, failure, chain, capsys):
    def unavailable(operation, args):
        if operation == "put":
            raise failure

    backup.hook = unavailable
    assert backup.invoke() == 3
    error = backup.status()["last_error"]
    assert error["transport"] == {"operation": "put", "chain": chain}
    assert "FAKE SECRET" not in json.dumps(error) + str(capsys.readouterr())


@pytest.mark.parametrize("operation", ["put", "get", "ls"])
def test_installed_http_503_preserves_only_available_cause(backup, operation, capsys):
    calls = []
    backup.cli = lambda op, args: op == operation
    backup.connection = lambda host: SimpleNamespace(path=None, c=TransferConnection(raw_error(503, "ServiceUnavailable"), calls))
    config = backup.home / ".config/living-memory/s3cmd-backup.conf"
    with config.open("a") as stream:
        stream.write("use_http_expect = False\n")
    assert backup.invoke() == (3 if operation == "put" else 1)
    error = backup.status()["last_error"]
    # Transfer methods replace 503 errors outside an except block; their
    # installed exception chains contain no underlying S3Error/status.
    roots = {"put": "S3UploadError", "get": "S3DownloadError"}
    expected = [{"class": roots[operation]}] if operation in roots else [{"class": "S3Error", "status": 503}]
    assert error["transport"]["chain"] == expected
    assert calls.count("PUT" if operation == "put" else "GET") == (8 if operation == "ls" else 1)
    assert "Timeout" not in error["message"]
    assert "FAKE SECRET" not in json.dumps(error) + str(capsys.readouterr())


def test_installed_listing_stdout_remains_parseable(backup):
    backup.state.mkdir(parents=True)
    namespace = runpy.run_path(str(SCRIPT))
    backup.cli = lambda op, args: op == "ls"
    response = RawResponse(data=b"""<ListBucketResult>
        <IsTruncated>false</IsTruncated><Contents><Key>global.sqlite3.gz</Key>
        <LastModified>2026-01-01T00:00:00.000Z</LastModified><Size>1</Size>
        <ETag>0123456789abcdef0123456789abcdef</ETag></Contents></ListBucketResult>""")
    calls = []
    backup.connection = lambda host: SimpleNamespace(path=None, c=TransferConnection(response, calls))
    with (backup.state / "backup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert namespace["objects"]() == {FINAL}
    assert calls == ["GET"]


def test_read_only_diagnosis_preserves_unknown_status_and_remote_copies(backup, tmp_path):
    def failed(operation, args):
        if operation == "put":
            raise subprocess.TimeoutExpired("FAKE SECRET", 3600)

    backup.hook = failed
    assert backup.invoke() == 3
    before = (backup.state / "status.json").read_bytes()
    copies = dict(backup.objects)
    backup.calls.clear()
    backup.hook = lambda *args: None
    namespace = runpy.run_path(str(SCRIPT))
    with (backup.state / "backup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed = namespace["verify_download"](FINAL, tmp_path, "observed")
        assert namespace["objects"]() == set(copies)
        assert namespace["multipart"]() == []
    assert observed == backup.status()["transaction"]["previous"]
    assert {op for op, _ in backup.calls} == {"info", "get", "ls", "multipart"}
    assert (backup.state / "status.json").read_bytes() == before
    assert backup.objects == copies


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
    fresh = backup.objects[FINAL]
    backup.hook = lambda *args: None
    backup.calls.clear()
    assert backup.invoke() == 0
    assert not any(op in {"cp", "put"} for op, _ in backup.calls)
    assert backup.objects == {FINAL: fresh}
    assert backup.status()["result"] == "verified"


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
        backup.hook = lambda *args: None
        backup.calls.clear()
        assert backup.invoke() == 0
        assert not any(op in {"abortmp", "cp", "put", "get"} for op, _ in backup.calls)


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
    attempts = 8 if response.status in {408, 429, 500, 502, 503, 504} else 1
    assert backup.raw_calls == [("versioning", BUCKET)] * attempts
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
    attempts = 8 if response.status in {408, 429, 500, 502, 503, 504} else 1
    assert sum(op == "policy" for op, _ in backup.raw_calls) == attempts + (target != BUCKET)
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
    assert failed == 8
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
    assert backup.calls == [("info", (BUCKET,))] * (8 if failure == "deadline" else 1)
    assert backup.objects == {FINAL: backup.old}
    assert "FAKE SECRET" not in json.dumps(backup.status()) + str(capsys.readouterr())
