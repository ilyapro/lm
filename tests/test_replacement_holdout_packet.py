"""Adversarial contract tests for the independent replacement validator.

The builder and verifier are always exercised as isolated subprocesses.  The
test oracle below deliberately reimplements selection, split bucketing, and
identity HMACs; it never imports either production module or helper.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import hmac
import importlib.util
import json
import os
import random
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKET_ROOT = ROOT / "artifacts/animal-planet"
REPLACEMENT_ROOT = PACKET_ROOT / "evaluation/replacement-holdout"
BUILD = REPLACEMENT_ROOT / "recipe/build.py"
VERIFY = REPLACEMENT_ROOT / "recipe/verify.py"
STAGING = Path("/home/sfx/.cache/ap-audit/staging")
SNAPSHOT = Path("/tmp/lm-shadow-eval.O3vX3k/global.sqlite3")
ORIGINAL_MANIFEST = PACKET_ROOT / "manifest.json"
SPLITS = PACKET_ROOT / "corpus/splits.json"
START = "2026-08-12T23:13:24Z"
END = "2026-08-13T20:16:51Z"
SCOPES = ("project:ae", "project:online")
TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")

PUBLIC_FAILURE_CHECKS = frozenset(
    {
        "aggregate",
        "arguments",
        "boundary",
        "candidate_changed",
        "canonical",
        "classification",
        "containment",
        "disjointness",
        "extra_file",
        "hash",
        "internal",
        "manifest",
        "membership",
        "ordering",
        "original_packet",
        "plaintext",
        "privacy",
        "schema",
        "secret_material",
        "selection",
        "snapshot",
        "source",
        "source_changed",
        "source_pin",
        "source_schema",
        "source_size",
        "token",
        "token_collision",
        "token_mismatch",
    }
)

EXPECTED_RECEIPT = {
    "schema_version": 1,
    "mode": "keyed-preseal",
    "status": "pass",
    "mismatches": 0,
    "token_collisions": 0,
    "semantic_reads": 0,
    "supplement_events": 769,
    "automatic_events": 332,
    "organic_events": 437,
    "project_ae_events": 502,
    "project_online_events": 267,
    "direct_nodes": 619,
    "nodes": 804,
    "dev_automatic_events": 732,
    "dev_unique_tokens": 353,
    "repeated_automatic_families": 18,
    "repeated_automatic_events": 238,
    "unseen_in_dev_families": 16,
    "unseen_in_dev_events": 113,
}


class _SecretKey(bytearray):
    def __repr__(self) -> str:
        return "<redacted-key>"


@dataclass(repr=False)
class _KeyedFixture:
    draft: Path
    key: _SecretKey

    def __iter__(self):
        yield self.draft
        yield self.key

    def __repr__(self) -> str:
        return "<keyed-fixture>"


def _json_line(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def _json_document(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def _token(key: bytes, query: str, requested_scope: str) -> str:
    message = (" ".join(query.split()) + "\n" + requested_scope).encode()
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def _bucket(event_id: str) -> int:
    return int.from_bytes(hashlib.sha256(event_id.encode()).digest()[:8], "big") % 100


def _raw_events(snapshot: Path = SNAPSHOT) -> list[dict]:
    connection = sqlite3.connect(f"file:{snapshot}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT * FROM recall_events WHERE created_at > ? AND created_at < ? "
            "AND requested_scope IN (?,?) ORDER BY created_at,id",
            (START, END, *SCOPES),
        ).fetchall()
    finally:
        connection.close()
    result = []
    for row in rows:
        value = dict(row)
        for name, expected in (("resolved_scopes", list), ("ambient_context", dict), ("results", list)):
            value[name] = json.loads(value[name])
            assert type(value[name]) is expected
        result.append(value)
    return result


def _staging_rows() -> list[dict]:
    return [json.loads(line) for line in (STAGING / "alt-db/recall_events.jsonl").read_text().splitlines()]


def _read_corpus(packet_or_draft: Path) -> list[dict]:
    return [json.loads(line) for line in (packet_or_draft / "corpus/holdout.jsonl").read_text().splitlines()]


def _write_corpus(packet_or_draft: Path, rows: list[dict]) -> None:
    (packet_or_draft / "corpus/holdout.jsonl").write_text("".join(_json_line(row) for row in rows))


def _read_index(packet_or_draft: Path) -> dict:
    return json.loads((packet_or_draft / "corpus/dev-fingerprint-index.json").read_text())


def _write_index(packet_or_draft: Path, value: dict) -> None:
    (packet_or_draft / "corpus/dev-fingerprint-index.json").write_text(_json_document(value))


def _make_writable(path: Path) -> None:
    for candidate in [path, *path.rglob("*")]:
        if not candidate.is_symlink():
            candidate.chmod(candidate.stat().st_mode | 0o200)


def _rekey_draft(draft: Path, key: bytes) -> None:
    raw_by_id = {row["id"]: row for row in _raw_events()}
    rows = _read_corpus(draft)
    for row in rows:
        if row["type"] == "event":
            raw = raw_by_id[row["id"]]
            row["fingerprint_token"] = _token(key, raw["query"], raw["requested_scope"])
    _write_corpus(draft, rows)
    dev_tokens = []
    for row in _staging_rows():
        if 15 <= _bucket(row["id"]) < 66 and row["agent"] is None:
            dev_tokens.append(_token(key, row["query"], row["requested_scope"]))
    index = _read_index(draft)
    index["tokens"] = sorted(set(dev_tokens))
    index["unique_tokens"] = len(index["tokens"])
    _write_index(draft, index)


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        cwd=ROOT,
        input=kwargs.pop("input", None),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=kwargs.pop("timeout", 120),
        check=False,
        **kwargs,
    )


def _assert_no_private_blob(output: bytes, secrets_: list[bytes | bytearray]) -> None:
    for secret in secrets_:
        secret = bytes(secret)
        variants = {
            secret,
            secret.hex().encode(),
            secret.hex().upper().encode(),
            base64.b64encode(secret),
            base64.urlsafe_b64encode(secret),
        }
        if any(value and value in output for value in variants):
            pytest.fail("private output leak", pytrace=False)


def _assert_no_private_output(
    result: subprocess.CompletedProcess[bytes], secrets_: list[bytes | bytearray]
) -> None:
    _assert_no_private_blob(result.stdout + result.stderr, secrets_)


def _assert_keyed_silent(result: subprocess.CompletedProcess[bytes]) -> None:
    if result.stdout or result.stderr:
        pytest.fail("keyed verifier emitted output", pytrace=False)


def _assert_empty_receipt(receipt: bytes) -> None:
    if receipt:
        pytest.fail("keyed verifier emitted a failure receipt", pytrace=False)


def _assert_keyed_pass(
    result: subprocess.CompletedProcess[bytes],
    receipt: bytes,
    *,
    private_values: list[bytes | bytearray] | None = None,
) -> None:
    private_values = private_values or []
    _assert_no_private_output(result, private_values)
    _assert_no_private_blob(receipt, private_values)
    _assert_keyed_silent(result)
    if result.returncode != 0 or not receipt or len(receipt) >= 1024:
        pytest.fail("keyed verifier did not emit one aggregate pass receipt", pytrace=False)
    try:
        payload = json.loads(receipt)
    except (UnicodeDecodeError, json.JSONDecodeError):
        pytest.fail("keyed verifier receipt is not aggregate JSON", pytrace=False)
    if payload != EXPECTED_RECEIPT:
        pytest.fail("keyed verifier receipt does not match the aggregate contract", pytrace=False)


def _assert_keyed_rejection(
    result: subprocess.CompletedProcess[bytes],
    receipt: bytes,
    *,
    private_values: list[bytes | bytearray] | None = None,
) -> None:
    _assert_no_private_output(result, private_values or [])
    if result.returncode == 0:
        pytest.fail("keyed verifier accepted an adversarial mutation", pytrace=False)
    _assert_empty_receipt(receipt)
    _assert_keyed_silent(result)


def _assert_sealed_attestation(
    result: subprocess.CompletedProcess[bytes],
    *,
    expected_status: str,
    private_values: list[bytes | bytearray] | None = None,
) -> dict:
    _assert_no_private_output(result, private_values or [])
    if result.stderr:
        pytest.fail("sealed verifier emitted stderr", pytrace=False)
    if not result.stdout or len(result.stdout) >= 512:
        pytest.fail("sealed verifier emitted a non-aggregate attestation", pytrace=False)
    try:
        payload = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError):
        pytest.fail("sealed verifier emitted invalid aggregate JSON", pytrace=False)
    if type(payload) is not dict or payload.get("status") != expected_status:
        pytest.fail("sealed verifier emitted the wrong aggregate status", pytrace=False)
    if expected_status == "pass":
        expected = {"mode": "keyless-sealed", "semantic_reads": 0, "status": "pass"}
        if payload != expected or result.returncode != 0:
            pytest.fail("sealed verifier pass attestation is not exact", pytrace=False)
    else:
        if (
            result.returncode == 0
            or set(payload) != {"mode", "status", "check"}
            or payload.get("mode") != "keyless-sealed"
            or payload.get("check") not in PUBLIC_FAILURE_CHECKS
        ):
            pytest.fail("sealed verifier failure attestation is not aggregate-only", pytrace=False)
    return payload


def _run_keyed(draft: Path, key: bytes, *, verifier: Path = VERIFY, snapshot: Path = SNAPSHOT) -> tuple[subprocess.CompletedProcess[bytes], bytes]:
    key_read, key_write = os.pipe()
    receipt_read, receipt_write = os.pipe()
    draft_fd = os.open(draft, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.write(key_write, key)
        os.close(key_write)
        key_write = -1
        result = _run(
            [
                sys.executable, "-I", os.fspath(verifier), "keyed",
                "--staging", os.fspath(STAGING),
                "--snapshot", os.fspath(snapshot),
                "--original-manifest", os.fspath(ORIGINAL_MANIFEST),
                "--splits", os.fspath(SPLITS),
                "--draft-dir", os.fspath(draft),
                "--draft-dir-fd", str(draft_fd),
                "--identity-key-fd", str(key_read),
                "--receipt-fd", str(receipt_write),
            ],
            pass_fds=(draft_fd, key_read, receipt_write),
        )
        os.close(receipt_write)
        receipt_write = -1
        receipt = b""
        while True:
            chunk = os.read(receipt_read, 4096)
            if not chunk:
                break
            receipt += chunk
        return result, receipt
    finally:
        for fd in (key_read, key_write, receipt_read, receipt_write, draft_fd):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass


def _run_sealed(packet: Path, *, original_manifest: Path = ORIGINAL_MANIFEST) -> subprocess.CompletedProcess[bytes]:
    return _run([
        sys.executable, "-I", os.fspath(VERIFY), "sealed",
        "--packet-dir", os.fspath(packet),
        "--original-manifest", os.fspath(original_manifest),
    ])


@pytest.fixture(scope="session")
def keyed_base(tmp_path_factory: pytest.TempPathFactory) -> _KeyedFixture:
    root = tmp_path_factory.mktemp("replacement-keyed")
    draft = root / "draft"
    built = _run([sys.executable, os.fspath(BUILD), "draft", "--draft-dir", os.fspath(draft)])
    assert built.returncode == 0
    _make_writable(draft)
    (draft / "manifest.json").unlink()
    key = _SecretKey(secrets.token_bytes(32))
    _rekey_draft(draft, key)
    return _KeyedFixture(draft, key)


@pytest.fixture(scope="session")
def sealed_base(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("replacement-sealed")
    publish = root / "packet"
    (publish / "recipe").mkdir(parents=True)
    shutil.copy2(BUILD, publish / "recipe/build.py")
    shutil.copy2(VERIFY, publish / "recipe/verify.py")
    shutil.copy2(REPLACEMENT_ROOT / "README.md", publish / "README.md")
    shutil.copy2(REPLACEMENT_ROOT / "POLICY.md", publish / "POLICY.md")
    digest = hashlib.sha256(VERIFY.read_bytes()).hexdigest()
    result = _run([
        sys.executable, os.fspath(BUILD), "freeze",
        "--draft-dir", os.fspath(root / "draft"),
        "--publish-dir", os.fspath(publish),
        "--verifier-sha256", digest,
    ])
    assert result.returncode == 0
    return publish


def _copy_fixture(source: Path, target: Path) -> Path:
    shutil.copytree(source, target)
    _make_writable(target)
    return target


def _rehash_data_file(packet: Path, relative: str) -> None:
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    raw = (packet / relative).read_bytes()
    metadata = manifest["files"][relative]
    metadata["sha256"] = hashlib.sha256(raw).hexdigest()
    metadata["bytes"] = len(raw)
    if relative.endswith("holdout.jsonl"):
        rows = [json.loads(line) for line in raw.splitlines()]
        metadata["records"] = len(rows)
        metadata["nodes"] = sum(row.get("type") == "node" for row in rows)
        metadata["events"] = sum(row.get("type") == "event" for row in rows)
    else:
        index = json.loads(raw)
        metadata["population_events"] = index.get("population_events")
        metadata["unique_tokens"] = index.get("unique_tokens")
    manifest_path.write_text(_json_document(manifest))


def _different_pure_letter(value: str) -> str:
    """Change one non-whitespace surrogate position without changing shape."""

    position = next(index for index, character in enumerate(value) if character not in " \t\n\r")
    replacement = "b" if value[position] == "a" else "a"
    return value[:position] + replacement + value[position + 1 :]


def test_files_exist_and_verifier_does_not_import_builder_helpers() -> None:
    assert VERIFY.stat().st_size > 0
    assert Path(__file__).stat().st_size > 0
    source = VERIFY.read_text()
    tree = ast.parse(source)
    forbidden_modules = {"build", "deid", "importlib", "runpy"}
    forbidden_calls = {"__import__", "compile", "eval", "exec"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(set(alias.name.split(".")) & forbidden_modules for alias in node.names):
                pytest.fail("verifier imports a forbidden helper or dynamic loader", pytrace=False)
        elif isinstance(node, ast.ImportFrom):
            module_parts = set((node.module or "").split("."))
            imported_parts = {alias.name for alias in node.names}
            if (module_parts | imported_parts) & forbidden_modules:
                pytest.fail("verifier imports a forbidden helper or dynamic loader", pytrace=False)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
                pytest.fail("verifier uses a dynamic code-loading primitive", pytrace=False)
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "exec_module",
                "load_module",
                "run_module",
                "run_path",
                "spec_from_file_location",
            }:
                pytest.fail("verifier uses a dynamic module-loading primitive", pytrace=False)


def test_source_blind_replacement_docs_authorize_only_reader() -> None:
    readme = (REPLACEMENT_ROOT / "README.md").read_text()
    policy = (REPLACEMENT_ROOT / "POLICY.md").read_text()

    def normalized(document: str) -> str:
        return " ".join(document.replace("`", "").replace("*", "").split()).lower()

    readme_contract = normalized(readme)
    policy_contract = normalized(policy)
    for contract in (readme_contract, policy_contract):
        assert contract.count("replacement-holdout-eval") == 1
        assert "first and only authorized semantic reader" in contract
        assert "holdout-shadow-eval" not in contract
        assert "semantic_reads: 0" in contract
        assert (
            "synthetic" in contract
            and "development" in contract
            and "evaluation" in contract
        )

    assert "exactly once" in readme_contract
    assert (
        "failed or partial semantic attempt consumes the single authorization"
        in readme_contract
    )
    assert "must not make or authorize a second semantic read" in policy_contract
    assert "does not restore the authorization" in policy_contract
    assert "without reopening the packet" in policy_contract


def test_source_blind_replacement_docs_match_process_lifetime_boundary() -> None:
    readme_contract = " ".join(
        (REPLACEMENT_ROOT / "README.md").read_text().replace("`", "").split()
    ).lower()
    policy_contract = " ".join(
        (REPLACEMENT_ROOT / "POLICY.md").read_text().replace("`", "").split()
    ).lower()
    for contract in (readme_contract, policy_contract):
        assert "process exit is the hard destruction boundary" in contract
        assert "mutable" in contract and "defense-in-depth" in contract
        assert "descriptor" in contract and "terminated or closed" in contract
    assert "before manifest construction or publication" in readme_contract
    assert (
        "manifest construction and manifest-last publication must occur only after that boundary"
        in policy_contract
    )

    def function(path: Path, name: str) -> tuple[str, ast.FunctionDef]:
        source = path.read_text()
        tree = ast.parse(source)
        node = next(
            candidate
            for candidate in tree.body
            if isinstance(candidate, ast.FunctionDef) and candidate.name == name
        )
        return source, node

    def call_name(call: ast.Call) -> str:
        value: ast.expr = call.func
        parts: list[str] = []
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))

    def call_line(node: ast.FunctionDef, name: str, argument: str | None = None) -> int:
        matches = []
        for candidate in ast.walk(node):
            if not isinstance(candidate, ast.Call) or call_name(candidate) != name:
                continue
            if argument is not None and not (
                candidate.args
                and isinstance(candidate.args[0], ast.Name)
                and candidate.args[0].id == argument
            ):
                continue
            matches.append(candidate.lineno)
        assert matches, f"missing lifecycle call {name}({argument or ''})"
        return min(matches)

    _external_source, external = function(BUILD, "_run_external_verifier")
    assert call_line(external, "_write_pipe_all") < call_line(
        external, "os.close", "key_write"
    ) < call_line(external, "process.wait") < call_line(
        external, "_validate_receipt"
    )

    candidate_source, candidate = function(BUILD, "_build_candidate")
    candidate_segment = ast.get_source_segment(candidate_source, candidate)
    assert candidate_segment is not None
    assert (
        "identity_key = bytearray(os.urandom(IDENTITY_KEY_BYTES))"
        in candidate_segment
    )
    assert (
        "surrogate_salt = bytearray(os.urandom(SURROGATE_SALT_BYTES))"
        in candidate_segment
    )
    assert (
        "for index in range(len(identity_key)):\n            identity_key[index] = 0"
        in candidate_segment
    )
    assert (
        "for index in range(len(surrogate_salt)):\n            surrogate_salt[index] = 0"
        in candidate_segment
    )

    worker_source, worker = function(BUILD, "_run_keyed_worker")
    child_branch = next(
        statement
        for statement in worker.body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Compare)
        and ast.get_source_segment(worker_source, statement.test) == "pid == 0"
    )
    child_segment = ast.get_source_segment(worker_source, child_branch)
    assert child_segment is not None
    assert "os._exit(exit_code)" in child_segment
    assert "os.close(status_write)" not in child_segment
    assert call_line(worker, "_read_pipe_limited") < call_line(
        worker, "os.killpg"
    ) < call_line(worker, "os.waitpid")

    _main_source, build_main = function(BUILD, "main")
    assert call_line(build_main, "_run_keyed_worker") < call_line(
        build_main, "_rehash_candidate"
    ) < call_line(build_main, "_manifest_document") < call_line(
        build_main, "_publish_validated"
    )

    key_source, read_key = function(VERIFY, "_read_exact_key")
    read_key_segment = ast.get_source_segment(key_source, read_key)
    assert read_key_segment is not None
    assert "os.close(fd)" in read_key_segment
    keyed_source, keyed = function(VERIFY, "_verify_keyed")
    keyed_segment = ast.get_source_segment(keyed_source, keyed)
    assert keyed_segment is not None
    assert (
        "for index_position in range(len(key)):\n            key[index_position] = 0"
        in keyed_segment
    )
    assert "os.close(args.receipt_fd)" in keyed_segment


def test_source_blind_replacement_document_pins_match_verifier() -> None:
    expected = {
        "README.md": (
            "c0483824717fb3323600b94db945c4ed8718e23749863f493f3013ecd2fae409",
            11_153,
        ),
        "POLICY.md": (
            "05d4f3d148b86a933c107341ac1157487509efe011428dc4858c18b9b1c28a90",
            6_958,
        ),
    }
    for name, pin in expected.items():
        raw = (REPLACEMENT_ROOT / name).read_bytes()
        assert (hashlib.sha256(raw).hexdigest(), len(raw)) == pin

    verifier_source = VERIFY.read_text()
    verifier_tree = ast.parse(verifier_source)
    assignments = {
        target.id: ast.literal_eval(statement.value)
        for statement in verifier_tree.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance((target := statement.targets[0]), ast.Name)
        and isinstance(statement.value, ast.Constant)
    }
    assert (assignments["README_SHA256"], assignments["README_BYTES"]) == expected[
        "README.md"
    ]
    assert (assignments["POLICY_SHA256"], assignments["POLICY_BYTES"]) == expected[
        "POLICY.md"
    ]
    validator = next(
        candidate
        for candidate in verifier_tree.body
        if isinstance(candidate, ast.FunctionDef)
        and candidate.name == "_validate_sealed_manifest"
    )
    used_names = {node.id for node in ast.walk(validator) if isinstance(node, ast.Name)}
    assert {
        "README_SHA256",
        "README_BYTES",
        "POLICY_SHA256",
        "POLICY_BYTES",
    } <= used_names

    test_source = Path(__file__).read_text()
    test_tree = ast.parse(test_source)
    fixture = next(
        candidate
        for candidate in test_tree.body
        if isinstance(candidate, ast.FunctionDef) and candidate.name == "sealed_base"
    )
    fixture_segment = ast.get_source_segment(test_source, fixture)
    assert fixture_segment is not None
    assert 'REPLACEMENT_ROOT / "README.md"' in fixture_segment
    assert 'REPLACEMENT_ROOT / "POLICY.md"' in fixture_segment
    assert 'PACKET_ROOT / "RECIPE.md"' not in fixture_segment
    assert 'PACKET_ROOT / "corpus/POLICY.md"' not in fixture_segment


def test_source_blind_mechanical_only_is_strict_sealed_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "_replacement_holdout_source_blind_verify", VERIFY
    )
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)

    packet_dir = REPLACEMENT_ROOT.resolve()
    original_manifest = PACKET_ROOT.resolve() / "manifest.json"
    shorthand = verifier._parse_args(["--mechanical-only"])
    monkeypatch.setattr(sys, "argv", [os.fspath(VERIFY), "--mechanical-only"])
    command_line = verifier._parse_args(None)
    explicit = verifier._parse_args(
        [
            "sealed",
            "--packet-dir",
            os.fspath(packet_dir),
            "--original-manifest",
            os.fspath(original_manifest),
        ]
    )
    assert vars(command_line) == vars(shorthand) == vars(explicit) == {
        "mode": "sealed",
        "packet_dir": packet_dir,
        "original_manifest": original_manifest,
    }
    with pytest.raises(verifier.VerificationError):
        verifier._parse_args(
            ["--mechanical-only", "--packet-dir", os.fspath(packet_dir)]
        )

    source = VERIFY.read_text()
    tree = ast.parse(source)
    main = next(
        candidate
        for candidate in tree.body
        if isinstance(candidate, ast.FunctionDef) and candidate.name == "main"
    )
    main_segment = ast.get_source_segment(source, main)
    assert main_segment is not None and "--mechanical-only" not in main_segment
    sealed_calls = [
        candidate
        for candidate in ast.walk(main)
        if isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
        and candidate.func.id == "_verify_sealed"
    ]
    assert len(sealed_calls) == 1


def test_independent_oracle_proves_dev_and_raw_aggregate_contract() -> None:
    rows = _staging_rows()
    dev = [row for row in rows if 15 <= _bucket(row["id"]) < 66]
    automatic = [row for row in dev if row["agent"] is None]
    all_automatic = [row for row in rows if row["agent"] is None]
    assert (len(dev), len(automatic), len(dev) - len(automatic)) == (826, 732, 94)
    assert any(_bucket(row["id"]) == 15 for row in automatic)
    assert any(_bucket(row["id"]) == 65 for row in automatic)
    assert any(_bucket(row["id"]) == 14 for row in all_automatic)
    assert any(_bucket(row["id"]) == 66 for row in all_automatic)
    assert all(_bucket(row["id"]) not in (14, 66) for row in automatic)
    selected = _raw_events()
    assert len(selected) == 769
    assert sum(row["agent"] is None for row in selected) == 332
    assert sum(row["agent"] is not None for row in selected) == 437
    assert {scope: sum(row["requested_scope"] == scope for row in selected) for scope in SCOPES} == {
        "project:ae": 502,
        "project:online": 267,
    }
    assert any(row["feedback_applied"] for row in selected)
    assert any(not row["feedback_applied"] for row in selected)


def test_keyed_accepts_independently_rekeyed_candidate_and_emits_exact_receipt(keyed_base: _KeyedFixture) -> None:
    draft, key = keyed_base
    result, receipt = _run_keyed(draft, key)
    _assert_keyed_pass(result, receipt, private_values=[key])


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_key",
        "short_key",
        "start_boundary",
        "end_boundary",
        "normalization",
        "event_token",
        "token_collision",
        "dev_token",
        "dev_order",
        "scope",
        "outcome",
        "class",
        "session",
        "unicode_whitespace",
        "surrogate_length",
        "surrogate_count",
        "ascii_whitespace",
        "equality_split",
        "bijection_merge",
        "privacy_ngram",
        "nested_text",
        "membership",
        "membership_unique",
        "node_containment",
        "extra_file",
    ],
)
def test_keyed_randomized_adversarial_mutations_reject(
    keyed_base: _KeyedFixture, tmp_path: Path, mutation: str
) -> None:
    base, key = keyed_base
    draft = _copy_fixture(base, tmp_path / "draft")
    rows = _read_corpus(draft)
    events = [row for row in rows if row["type"] == "event"]
    rng = random.SystemRandom()
    run_key = key
    if mutation == "wrong_key":
        run_key = _SecretKey(secrets.token_bytes(32))
    elif mutation == "short_key":
        run_key = _SecretKey(key[:-1])
    elif mutation in ("start_boundary", "end_boundary"):
        event = rng.choice(events)
        event["created_at"] = START if mutation == "start_boundary" else END
        _write_corpus(draft, rows)
    elif mutation == "normalization":
        raw_by_id = {row["id"]: row for row in _raw_events()}
        event = next(row for row in events if "\n" in raw_by_id[row["id"]]["query"])
        raw = raw_by_id[event["id"]]
        wrong_message = (raw["query"] + "\n" + raw["requested_scope"]).encode()
        event["fingerprint_token"] = hmac.new(key, wrong_message, hashlib.sha256).hexdigest()
        _write_corpus(draft, rows)
    elif mutation == "event_token":
        event = rng.choice(events)
        event["fingerprint_token"] = ("0" if event["fingerprint_token"][0] != "0" else "1") + event["fingerprint_token"][1:]
        _write_corpus(draft, rows)
    elif mutation == "token_collision":
        left = rng.choice(events)
        right = next(row for row in events if row["fingerprint_token"] != left["fingerprint_token"])
        right["fingerprint_token"] = left["fingerprint_token"]
        _write_corpus(draft, rows)
    elif mutation == "dev_token":
        index = _read_index(draft)
        index["tokens"][rng.randrange(len(index["tokens"]))] = "f" * 64
        index["tokens"] = sorted(set(index["tokens"]))
        index["unique_tokens"] = len(index["tokens"])
        _write_index(draft, index)
    elif mutation == "dev_order":
        index = _read_index(draft)
        index["tokens"][0], index["tokens"][1] = index["tokens"][1], index["tokens"][0]
        _write_index(draft, index)
    elif mutation == "scope":
        left = next(row for row in events if row["requested_scope"] == SCOPES[0])
        right = next(row for row in events if row["requested_scope"] == SCOPES[1])
        left["requested_scope"], right["requested_scope"] = right["requested_scope"], left["requested_scope"]
        _write_corpus(draft, rows)
    elif mutation == "outcome":
        event = next(row for row in events if row["feedback_applied"])
        event["feedback_applied"] = 0
        _write_corpus(draft, rows)
    elif mutation == "class":
        automatic = next(row for row in events if row["class"] == "automatic")
        organic = next(row for row in events if row["class"] == "organic")
        automatic["class"], organic["class"] = organic["class"], automatic["class"]
        _write_corpus(draft, rows)
    elif mutation == "session":
        groups: dict[str, list[dict]] = {}
        for row in events:
            if row["class"] == "automatic":
                groups.setdefault(row["fingerprint_token"], []).append(row)
        family = next(value for value in groups.values() if len(value) >= 3 and len({row["transport_session_id"] for row in value}) >= 2)
        for row in family:
            row["transport_session_id"] = family[0]["transport_session_id"]
        _write_corpus(draft, rows)
    elif mutation == "unicode_whitespace":
        event = next(row for row in events if row["query_surrogate"])
        event["query_surrogate"] = "\u2028" + event["query_surrogate"][1:]
        _write_corpus(draft, rows)
    elif mutation == "surrogate_length":
        event = next(row for row in events if row["query_surrogate"])
        event["query_surrogate"] = event["query_surrogate"][:-1]
        _write_corpus(draft, rows)
    elif mutation == "surrogate_count":
        event = rng.choice(events)
        event["query_chars"] += 1
        _write_corpus(draft, rows)
    elif mutation == "ascii_whitespace":
        raw_by_id = {row["id"]: row for row in _raw_events()}
        event = next(
            row for row in events
            if any(character in " \t\n\r" for character in raw_by_id[row["id"]]["query"])
        )
        original = raw_by_id[event["id"]]["query"]
        position = next(index for index, character in enumerate(original) if character in " \t\n\r")
        event["query_surrogate"] = (
            event["query_surrogate"][:position]
            + "a"
            + event["query_surrogate"][position + 1 :]
        )
        _write_corpus(draft, rows)
    elif mutation == "equality_split":
        raw_by_id = {row["id"]: row for row in _raw_events()}
        by_query: dict[str, list[dict]] = {}
        for event in events:
            by_query.setdefault(raw_by_id[event["id"]]["query"], []).append(event)
        family = next(
            family for family in by_query.values()
            if len(family) >= 2
            and any(character not in " \t\n\r" for character in family[0]["query_surrogate"])
        )
        family[-1]["query_surrogate"] = _different_pure_letter(family[-1]["query_surrogate"])
        _write_corpus(draft, rows)
    elif mutation == "bijection_merge":
        raw_by_id = {row["id"]: row for row in _raw_events()}
        by_shape: dict[tuple[int, tuple[tuple[int, str], ...]], list[dict]] = {}
        for event in events:
            original = raw_by_id[event["id"]]["query"]
            shape = (
                len(original),
                tuple(
                    (index, character)
                    for index, character in enumerate(original)
                    if character in " \t\n\r"
                ),
            )
            by_shape.setdefault(shape, []).append(event)
        left, right = next(
            (left, right)
            for family in by_shape.values()
            for left in family
            for right in family
            if raw_by_id[left["id"]]["query"] != raw_by_id[right["id"]]["query"]
            and left["query_surrogate"] != right["query_surrogate"]
        )
        right["query_surrogate"] = left["query_surrogate"]
        _write_corpus(draft, rows)
    elif mutation == "privacy_ngram":
        raw_by_id = {row["id"]: row for row in _raw_events()}
        event, offset = next(
            (event, offset)
            for event in events
            for offset in range(max(0, len(raw_by_id[event["id"]]["query"]) - 11))
            if len(raw_by_id[event["id"]]["query"]) > 12
            and re.fullmatch(
                r"[a-z \t\n\r]{12}",
                raw_by_id[event["id"]]["query"][offset : offset + 12],
            )
            and sum(
                character not in " \t\n\r"
                for character in raw_by_id[event["id"]]["query"][offset : offset + 12]
            )
            >= 7
        )
        original_gram = raw_by_id[event["id"]]["query"][offset : offset + 12]
        event["query_surrogate"] = (
            event["query_surrogate"][:offset]
            + original_gram
            + event["query_surrogate"][offset + 12 :]
        )
        _write_corpus(draft, rows)
    elif mutation == "nested_text":
        raw_by_id = {row["id"]: row for row in _raw_events()}
        event = next(
            row for row in events
            if any(
                isinstance(value, str)
                and len(value) >= 12
                and re.fullmatch(r"[a-z \t\n\r]+", value)
                and isinstance(row["ambient_context_surrogate"].get(name), str)
                for name, value in raw_by_id[row["id"]]["ambient_context"].items()
            )
        )
        field = next(
            name for name, value in raw_by_id[event["id"]]["ambient_context"].items()
            if isinstance(value, str)
            and len(value) >= 12
            and re.fullmatch(r"[a-z \t\n\r]+", value)
            and isinstance(event["ambient_context_surrogate"].get(name), str)
        )
        event["ambient_context_surrogate"][field] = raw_by_id[event["id"]]["ambient_context"][field]
        _write_corpus(draft, rows)
    elif mutation == "membership":
        same = [row for row in events if row["class"] == events[0]["class"] and row["requested_scope"] == events[0]["requested_scope"]]
        victim = same[-1]
        rows.remove(victim)
        rows.append(dict(same[0]))
        _write_corpus(draft, rows)
    elif mutation == "membership_unique":
        selected_ids = {row["id"] for row in events}
        connection = sqlite3.connect(f"file:{SNAPSHOT}?mode=ro&immutable=1", uri=True)
        try:
            replacement_id = next(
                value
                for (value,) in connection.execute("SELECT id FROM recall_events ORDER BY id")
                if value not in selected_ids and re.fullmatch(r"[0-9A-HJKMNP-TV-Z]{26}", value)
            )
        finally:
            connection.close()
        rng.choice(events)["id"] = replacement_id
        _write_corpus(draft, rows)
    elif mutation == "node_containment":
        event = next(row for row in events if row["results"])
        event["results"][0]["node_id"] = "0" * 26
        _write_corpus(draft, rows)
    elif mutation == "extra_file":
        (draft / "corpus/unexpected.bin").write_bytes(b"aggregate-only")
    result, receipt = _run_keyed(draft, run_key)
    raw_query = _raw_events()[0]["query"].encode()
    try:
        _assert_keyed_rejection(
            result,
            receipt,
            private_values=[key, run_key, raw_query],
        )
    finally:
        shutil.rmtree(draft)
        if run_key is not key:
            run_key[:] = b"\0" * len(run_key)


@pytest.mark.parametrize("encoding", ["plaintext", "key_hex", "key_base64", "unkeyed_digest"])
def test_keyed_rejects_plaintext_and_forbidden_key_or_unkeyed_fingerprint_encodings(
    keyed_base: _KeyedFixture, tmp_path: Path, encoding: str
) -> None:
    base, key = keyed_base
    raw_rows = _raw_events()
    raw = next(row for row in raw_rows if row["query"])
    normalized = (" ".join(raw["query"].split()) + "\n" + raw["requested_scope"]).encode()
    draft = _copy_fixture(base, tmp_path / "draft")
    rows = _read_corpus(draft)
    event = next(row for row in rows if row.get("id") == raw["id"])
    if encoding == "plaintext":
        payload = raw["query"].encode()
        event["query_surrogate"] = raw["query"]
    elif encoding == "key_hex":
        payload = key.hex().encode()
        event["fingerprint_token"] = payload.decode()
    elif encoding == "key_base64":
        payload = base64.b64encode(key)
        event["query_surrogate"] = payload.decode()
        event["query_chars"] = len(event["query_surrogate"])
    else:
        payload = hashlib.sha256(normalized).hexdigest().encode()
        event["fingerprint_token"] = payload.decode()
    _write_corpus(draft, rows)
    try:
        result, receipt = _run_keyed(draft, key)
        _assert_keyed_rejection(
            result,
            receipt,
            private_values=[key, raw["query"].encode(), payload],
        )
    finally:
        shutil.rmtree(draft)
        payload = b""


def test_keyed_rejects_source_boundary_alias_and_preserves_sources(
    keyed_base: _KeyedFixture, tmp_path: Path
) -> None:
    draft, key = keyed_base
    sources = [STAGING / "METADATA.json", STAGING / "alt-db/recall_events.jsonl", SNAPSHOT, ORIGINAL_MANIFEST, SPLITS]
    before = [(path.stat(), hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources]
    positive, positive_receipt = _run_keyed(draft, key)
    _assert_keyed_pass(positive, positive_receipt, private_values=[key])
    after_positive = [(path.stat(), hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources]
    assert [(x.st_size, x.st_mtime_ns, digest) for x, digest in before] == [
        (x.st_size, x.st_mtime_ns, digest) for x, digest in after_positive
    ]
    alias = tmp_path / "draft-alias"
    alias.symlink_to(draft, target_is_directory=True)
    result, receipt = _run_keyed(alias, key)
    _assert_keyed_rejection(result, receipt, private_values=[key])
    after = [(path.stat(), hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources]
    assert [(x.st_size, x.st_mtime_ns, digest) for x, digest in before] == [
        (x.st_size, x.st_mtime_ns, digest) for x, digest in after
    ]


def test_keyed_rejects_persistent_key_and_receipt_descriptors(
    keyed_base: _KeyedFixture, tmp_path: Path
) -> None:
    draft, key = keyed_base
    key_path = tmp_path / "key-channel"
    receipt_path = tmp_path / "receipt-channel"
    key_fd = os.open(key_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    receipt_fd = os.open(receipt_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    draft_fd = os.open(draft, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.write(key_fd, key)
        os.lseek(key_fd, 0, os.SEEK_SET)
        result = _run(
            [
                sys.executable, "-I", os.fspath(VERIFY), "keyed",
                "--staging", os.fspath(STAGING),
                "--snapshot", os.fspath(SNAPSHOT),
                "--original-manifest", os.fspath(ORIGINAL_MANIFEST),
                "--splits", os.fspath(SPLITS),
                "--draft-dir", os.fspath(draft),
                "--draft-dir-fd", str(draft_fd),
                "--identity-key-fd", str(key_fd),
                "--receipt-fd", str(receipt_fd),
            ],
            pass_fds=(draft_fd, key_fd, receipt_fd),
        )
        _assert_keyed_silent(result)
        if result.returncode == 0:
            pytest.fail("keyed verifier accepted persistent channels", pytrace=False)
    finally:
        for fd in (key_fd, receipt_fd, draft_fd):
            try:
                os.close(fd)
            except OSError:
                pass
        key_path.write_bytes(b"\0" * 32)
        key_path.unlink()
        receipt_path.unlink()


def test_unicode_normalization_oracle_is_python_split_and_not_ascii_only() -> None:
    key = _SecretKey(secrets.token_bytes(32))
    for whitespace in ("\u00a0", "\u2028", "\u2029", "\u2003", "\t", "\r\n"):
        query = "alpha" + whitespace + "beta"
        assert " ".join(query.split()) == "alpha beta"
        assert _token(key, query, SCOPES[0]) == _token(key, "alpha beta", SCOPES[0])
    assert _token(key, "alpha\u00a0beta", SCOPES[0]) != hmac.new(
        key, ("alpha\u00a0beta\n" + SCOPES[0]).encode(), hashlib.sha256
    ).hexdigest()


def test_keyed_selection_uses_strict_endpoints_and_requested_scope(
    keyed_base: _KeyedFixture, tmp_path: Path
) -> None:
    """Decoys and an exotic-space row distinguish the exact raw predicate."""

    base_draft, key = keyed_base
    draft = _copy_fixture(base_draft, tmp_path / "unicode-draft")
    snapshot = tmp_path / "snapshot.sqlite3"
    copied = _run(["cp", "--reflink=auto", os.fspath(SNAPSHOT), os.fspath(snapshot)])
    assert copied.returncode == 0
    connection = sqlite3.connect(snapshot)
    try:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(recall_events)")]
        template = {
            "query": "synthetic boundary sentinel",
            "scope": SCOPES[0],
            "requested_scope": SCOPES[0],
            "resolved_scopes": json.dumps(list(SCOPES)),
            "ambient_context": "{}",
            "depth": "normal",
            "max_results": 10,
            "results": "[]",
            "agent": "synthetic",
            "task": None,
            "session_id": None,
            "feedback_applied": 0,
            "feedback_trace_id": None,
            "feedback_applied_at": None,
            "transport_session_id": "0" * 32,
        }
        decoys = []
        for suffix, created_at in (("A", START), ("B", END)):
            row = dict(template, id=("0" * 25) + suffix, created_at=created_at)
            decoys.append(tuple(row.get(column) for column in columns))
        wrong_scope = dict(
            template,
            id=("0" * 25) + "C",
            created_at="2026-08-13T10:00:00Z",
            requested_scope="project:outside",
        )
        decoys.append(tuple(wrong_scope.get(column) for column in columns))
        connection.executemany(
            f"INSERT INTO recall_events ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
            decoys,
        )
        selected = _raw_events()
        query_counts: dict[str, int] = {}
        for row in selected:
            query_counts[row["query"]] = query_counts.get(row["query"], 0) + 1
        unicode_row = next(
            row for row in selected
            if "\n" in row["query"] and query_counts[row["query"]] == 1
        )
        newline_position = unicode_row["query"].index("\n")
        unicode_query = (
            unicode_row["query"][:newline_position]
            + "\u2028"
            + unicode_row["query"][newline_position + 1 :]
        )
        if " ".join(unicode_query.split()) != " ".join(unicode_row["query"].split()):
            pytest.fail("unicode normalization oracle mismatch", pytrace=False)
        connection.execute(
            "UPDATE recall_events SET query=? WHERE id=?",
            (unicode_query, unicode_row["id"]),
        )
        connection.commit()
    finally:
        connection.close()
    corpus = _read_corpus(draft)
    event = next(row for row in corpus if row.get("id") == unicode_row["id"])
    assert event["query_surrogate"][newline_position] == "\n"
    event["query_surrogate"] = (
        event["query_surrogate"][:newline_position]
        + "a"
        + event["query_surrogate"][newline_position + 1 :]
    )
    _write_corpus(draft, corpus)
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    verifier = tmp_path / "verify.py"
    source = VERIFY.read_text()
    old = "4d6648f9e3c33e8ffaa7bf15620bd26a18bdf9f7ddf5b97641aa8c082e9aaa67"
    assert source.count(old) == 1
    verifier.write_text(source.replace(old, digest))
    result, receipt = _run_keyed(draft, key, verifier=verifier, snapshot=snapshot)
    _assert_keyed_pass(result, receipt, private_values=[key])


def test_sealed_accepts_actual_frozen_packet(sealed_base: Path) -> None:
    result = _run_sealed(sealed_base)
    first_token = next(
        row["fingerprint_token"].encode()
        for row in _read_corpus(sealed_base)
        if row["type"] == "event"
    )
    _assert_sealed_attestation(
        result,
        expected_status="pass",
        private_values=[_raw_events()[0]["query"].encode(), first_token],
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "stale_hash",
        "manifest_total",
        "frozen",
        "semantic_reads",
        "receipt_semantic_reads",
        "class_rehashed",
        "scope_rehashed",
        "arbitrary_scope_rehashed",
        "node_scope_rehashed",
        "resolved_scope_rehashed",
        "arbitrary_id_rehashed",
        "date_only_created_at_rehashed",
        "arbitrary_text_channel_rehashed",
        "temporal_hint_rehashed",
        "numeric_type_rehashed",
        "max_results_rehashed",
        "decay_coherence_rehashed",
        "temporal_order_rehashed",
        "provenance_kind_rehashed",
        "ambient_key_channel_rehashed",
        "provenance_key_channel_rehashed",
        "provenance_negative_chars_rehashed",
        "binary_flag_rehashed",
        "boolean_flag_rehashed",
        "session_family_rehashed",
        "session_text_rehashed",
        "node_containment_rehashed",
        "result_metadata_rehashed",
        "orphan_partner_rehashed",
        "context_chars_rehashed",
        "token_syntax_rehashed",
        "index_count_rehashed",
        "index_membership_rehashed",
        "nested_text_rehashed",
        "event_membership_rehashed",
        "original_dev_overlap_rehashed",
        "original_eval_overlap_rehashed",
        "original_holdout_overlap_rehashed",
        "schema_order_rehashed",
        "extra_root_file",
        "extra_corpus_file",
    ],
)
def test_sealed_randomized_hash_manifest_and_semantic_mutations_reject(
    sealed_base: Path, tmp_path: Path, mutation: str
) -> None:
    packet = _copy_fixture(sealed_base, tmp_path / "packet")
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = _read_corpus(packet)
    events = [row for row in rows if row["type"] == "event"]
    nodes = [row for row in rows if row["type"] == "node"]
    opaque_probe = events[0]["fingerprint_token"].encode()
    if mutation == "stale_hash":
        path = packet / "corpus/holdout.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
    elif mutation == "manifest_total":
        manifest["counts"]["supplement"]["events"] += 1
        manifest_path.write_text(_json_document(manifest))
    elif mutation == "frozen":
        manifest["frozen"] = False
        manifest_path.write_text(_json_document(manifest))
    elif mutation == "semantic_reads":
        manifest["semantic_reads"] = 1
        manifest_path.write_text(_json_document(manifest))
    elif mutation == "receipt_semantic_reads":
        manifest["validation"]["keyed_preseal"]["semantic_reads"] = 1
        manifest_path.write_text(_json_document(manifest))
    elif mutation == "class_rehashed":
        automatic = next(row for row in events if row["class"] == "automatic")
        organic = next(row for row in events if row["class"] == "organic")
        automatic["class"], organic["class"] = organic["class"], automatic["class"]
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "scope_rehashed":
        left = next(row for row in events if row["requested_scope"] == SCOPES[0])
        left["requested_scope"] = "project:outside"
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "arbitrary_scope_rehashed":
        events[0]["scope"] = "project:forged"
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "node_scope_rehashed":
        partner = next(node for node in nodes if node["included_via"] == ["typed_edge_partner"])
        partner["scope"] = "project:private-diagnostic"
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "resolved_scope_rehashed":
        events[0]["resolved_scopes"] = ["global", events[0]["requested_scope"]]
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "arbitrary_id_rehashed":
        events[0]["id"] = "invalid-event-identifier"
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "date_only_created_at_rehashed":
        event = next(row for row in events if row["created_at"].startswith("2026-08-13T"))
        event["created_at"] = "2026-08-13"
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "arbitrary_text_channel_rehashed":
        events[0]["depth"] = "private diagnostic channel"
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "temporal_hint_rehashed":
        node = next(row for row in nodes if row["stats"]["temporal_hint"] is not None)
        node["stats"]["temporal_hint"] = "private:diagnostic"
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "numeric_type_rehashed":
        event = next(row for row in events if row["results"])
        event["results"][0]["score"] = 999
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "max_results_rehashed":
        event = next(row for row in events if row["results"])
        event["max_results"] = 0
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "decay_coherence_rehashed":
        node = next(row for row in nodes if row["decayed"] == 1)
        node["decay_reason_surrogate"] = None
        node["decay_reason_chars"] = None
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "temporal_order_rehashed":
        event = next(row for row in events if row["feedback_applied_at"] is not None)
        event["feedback_applied_at"] = "2000-01-01T00:00:00Z"
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "provenance_kind_rehashed":
        node = next(row for row in nodes if row["provenance_shape"]["keys"])
        name = next(iter(node["provenance_shape"]["keys"]))
        node["provenance_shape"]["keys"][name] = {"kind": "NoneType", "chars": 0}
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "ambient_key_channel_rehashed":
        event = next(row for row in events if row["ambient_context_surrogate"])
        old = next(iter(event["ambient_context_surrogate"]))
        value = event["ambient_context_surrogate"].pop(old)
        count = event["ambient_context_chars"].pop(old)
        event["ambient_context_surrogate"]["private diagnostics"] = value
        event["ambient_context_chars"]["private diagnostics"] = count
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "provenance_key_channel_rehashed":
        node = next(row for row in nodes if row["provenance_shape"]["keys"])
        old = next(iter(node["provenance_shape"]["keys"]))
        value = node["provenance_shape"]["keys"].pop(old)
        node["provenance_shape"]["keys"]["private diagnostics"] = value
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "provenance_negative_chars_rehashed":
        node = next(row for row in nodes if row["provenance_shape"]["keys"])
        node["provenance_shape"]["serialized_chars"] = 0
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "binary_flag_rehashed":
        events[0]["feedback_applied"] = 2
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "boolean_flag_rehashed":
        events[0]["feedback_applied"] = False
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "session_family_rehashed":
        groups: dict[str, list[dict]] = {}
        for event in events:
            if event["class"] == "automatic":
                groups.setdefault(event["fingerprint_token"], []).append(event)
        family = next(
            family for family in groups.values()
            if len(family) >= 3
            and len({event["transport_session_id"] for event in family}) >= 2
        )
        session = family[0]["transport_session_id"]
        for event in family:
            event["transport_session_id"] = session
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "session_text_rehashed":
        organic = next(row for row in events if row["class"] == "organic")
        organic["transport_session_id"] = "private diagnostics"
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "node_containment_rehashed":
        event = next(row for row in events if row["results"])
        event["results"][0]["node_id"] = "0" * 26
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "result_metadata_rehashed":
        event = next(row for row in events if row["results"])
        event["results"][0]["level"] = (
            "schema" if event["results"][0]["level"] != "schema" else "trace"
        )
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "orphan_partner_rehashed":
        direct = {
            result["node_id"] for event in events for result in event["results"]
        } | {
            event["feedback_trace_id"]
            for event in events
            if event["feedback_trace_id"] is not None
        }
        node_by_id = {node["id"]: node for node in nodes}
        partner = next(
            node for node in nodes
            if node["id"] not in direct
            and any(relation["other_id"] in direct for relation in node["relations"])
        )
        for relation in list(partner["relations"]):
            if relation["other_id"] in direct:
                other = node_by_id[relation["other_id"]]
                other["relations"] = [
                    candidate for candidate in other["relations"]
                    if not (
                        candidate["other_id"] == partner["id"]
                        and candidate["type"] == relation["type"]
                        and candidate["created_at"] == relation["created_at"]
                    )
                ]
        partner["relations"] = [
            relation for relation in partner["relations"]
            if relation["other_id"] not in direct
        ]
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "context_chars_rehashed":
        node = next(row for row in nodes if row["context_chars"])
        name = next(iter(node["context_chars"]))
        node["context_chars"][name] += 1
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "token_syntax_rehashed":
        events[0]["fingerprint_token"] = events[0]["fingerprint_token"].upper()
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "index_count_rehashed":
        index = _read_index(packet)
        index["tokens"].append("0" * 64)
        index["tokens"] = sorted(index["tokens"])
        index["unique_tokens"] = len(index["tokens"])
        _write_index(packet, index)
        _rehash_data_file(packet, "corpus/dev-fingerprint-index.json")
    elif mutation == "index_membership_rehashed":
        index = _read_index(packet)
        unseen_repeated = next(
            token for token, family in {
                token: [row for row in events if row["class"] == "automatic" and row["fingerprint_token"] == token]
                for token in {row["fingerprint_token"] for row in events if row["class"] == "automatic"}
            }.items()
            if len(family) >= 3
            and len({row["transport_session_id"] for row in family}) >= 2
            and token not in index["tokens"]
        )
        automatic_tokens = {
            row["fingerprint_token"] for row in events if row["class"] == "automatic"
        }
        dropped = next(token for token in index["tokens"] if token not in automatic_tokens)
        index["tokens"].remove(dropped)
        index["tokens"].append(unseen_repeated)
        index["tokens"] = sorted(index["tokens"])
        _write_index(packet, index)
        _rehash_data_file(packet, "corpus/dev-fingerprint-index.json")
    elif mutation == "nested_text_rehashed":
        event = next(row for row in events if row["ambient_context_surrogate"])
        event["ambient_context_surrogate"]["private_diagnostic"] = "forbidden-diagnostic"
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "event_membership_rehashed":
        events[-1]["id"] = events[0]["id"]
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation in {
        "original_dev_overlap_rehashed",
        "original_eval_overlap_rehashed",
        "original_holdout_overlap_rehashed",
    }:
        split = mutation.removeprefix("original_").removesuffix("_overlap_rehashed")
        original_id = next(
            json.loads(line)["id"]
            for line in (PACKET_ROOT / f"corpus/{split}.jsonl").read_text().splitlines()
            if json.loads(line).get("type") == "event"
        )
        events[0]["id"] = original_id
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "schema_order_rehashed":
        event = events[0]
        value = event.pop("id")
        event["id"] = value
        _write_corpus(packet, rows)
        _rehash_data_file(packet, "corpus/holdout.jsonl")
    elif mutation == "extra_root_file":
        (packet / "unexpected.txt").write_text("aggregate-only")
    elif mutation == "extra_corpus_file":
        (packet / "corpus/unexpected.json").write_text("{}\n")
    result = _run_sealed(packet)
    _assert_sealed_attestation(
        result,
        expected_status="fail",
        private_values=[_raw_events()[0]["query"].encode(), opaque_probe],
    )


def test_sealed_rejects_original_packet_pin_or_hash_mutation(sealed_base: Path, tmp_path: Path) -> None:
    original_root = tmp_path / "original"
    shutil.copytree(PACKET_ROOT, original_root)
    _make_writable(original_root)
    original_manifest = original_root / "manifest.json"
    manifest = json.loads(original_manifest.read_text())
    manifest["frozen"] = False
    original_manifest.write_text(_json_document(manifest))
    result = _run_sealed(sealed_base, original_manifest=original_manifest)
    _assert_sealed_attestation(
        result,
        expected_status="fail",
        private_values=[_raw_events()[0]["query"].encode()],
    )


def test_sealed_rejects_canonical_json_and_packet_file_hash_changes(sealed_base: Path, tmp_path: Path) -> None:
    packet = _copy_fixture(sealed_base, tmp_path / "packet")
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    result = _run_sealed(packet)
    _assert_sealed_attestation(
        result,
        expected_status="fail",
        private_values=[_raw_events()[0]["query"].encode()],
    )
    packet = _copy_fixture(sealed_base, tmp_path / "packet-static")
    (packet / "README.md").write_text((packet / "README.md").read_text() + "\n")
    result = _run_sealed(packet)
    _assert_sealed_attestation(
        result,
        expected_status="fail",
        private_values=[_raw_events()[0]["query"].encode()],
    )


def test_sealed_rejects_rehashed_static_plaintext_channel(sealed_base: Path, tmp_path: Path) -> None:
    packet = _copy_fixture(sealed_base, tmp_path / "packet-static-rehashed")
    private = _raw_events()[0]["query"].encode()
    readme = packet / "README.md"
    readme.write_bytes(readme.read_bytes() + b"\n" + private + b"\n")
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    raw = readme.read_bytes()
    manifest["packet_files"]["README.md"] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    manifest_path.write_text(_json_document(manifest))
    result = _run_sealed(packet)
    try:
        _assert_sealed_attestation(result, expected_status="fail", private_values=[private])
    finally:
        readme.write_bytes(b"")


def test_sealed_binds_packet_verifier_to_executing_verifier(sealed_base: Path, tmp_path: Path) -> None:
    packet = _copy_fixture(sealed_base, tmp_path / "packet-verifier-rehashed")
    verifier = packet / "recipe/verify.py"
    verifier.write_text(verifier.read_text() + "\n# private diagnostic channel\n")
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    raw = verifier.read_bytes()
    metadata = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    manifest["packet_files"]["recipe/verify.py"] = metadata
    manifest["implementation"]["verifier"] = {
        "path": "recipe/verify.py",
        **metadata,
    }
    manifest_path.write_text(_json_document(manifest))
    result = _run_sealed(packet)
    _assert_sealed_attestation(result, expected_status="fail")


def test_randomized_family_oracle_matches_declared_aggregates(keyed_base: _KeyedFixture) -> None:
    draft, _key = keyed_base
    rows = _read_corpus(draft)
    events = [row for row in rows if row["type"] == "event" and row["class"] == "automatic"]
    groups: dict[str, list[dict]] = {}
    for row in events:
        groups.setdefault(row["fingerprint_token"], []).append(row)
    repeated = {
        token: family for token, family in groups.items()
        if len(family) >= 3 and len({row["transport_session_id"] for row in family}) >= 2
    }
    dev = set(_read_index(draft)["tokens"])
    unseen = {token: family for token, family in repeated.items() if token not in dev}
    assert (len(repeated), sum(map(len, repeated.values()))) == (18, 238)
    assert (len(unseen), sum(map(len, unseen.values()))) == (16, 113)
    rng = random.SystemRandom()
    sample = rng.sample(list(repeated.values()), k=min(5, len(repeated)))
    assert all(len(family) >= 3 and len({row["transport_session_id"] for row in family}) >= 2 for family in sample)


def test_keyless_output_contains_only_aggregate_attestation(sealed_base: Path) -> None:
    result = _run_sealed(sealed_base)
    output = result.stdout + result.stderr
    first_query = _raw_events()[0]["query"].encode()
    first_token = next(row["fingerprint_token"].encode() for row in _read_corpus(sealed_base) if row["type"] == "event")
    _assert_no_private_output(result, [first_query, first_token])
    assert len(output) < 256
