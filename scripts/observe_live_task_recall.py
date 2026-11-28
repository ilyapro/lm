#!/usr/bin/env python3
"""Compare a published LM candidate or observe recall after a reader update.

Use the compare subcommand for an independent disk and /health comparison.
Run with compare --help for options. The default invocation checks recall.

Put private cases in a file outside Git, for example
{"queries": [{"query": "...", "expected_node_id": "..."}, ...]}.
The file is read locally; neither it nor recall responses enter the receipt.
Provide the boot ID and loaded-code digest collected before the update.

``--native-completion`` emits the fresh JSON projection expected by AE's native
verify_command. It is local-only and authenticates /admin/info, compares the live
function digest with an isolated import of the published commit, and checks the
installed Python source bytes and their pre-process timestamps. ``loaded_bytes``
is explicitly a published-source projection supported by that function identity;
the existing health endpoint does not attest module globals or original source.
Private cases and recall responses never enter either output format.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import ipaddress
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.request import Request, urlopen

from check_deployed_protocol import build_url, build_verify, resolve_token

CODE_IDENTITY_SCHEME = "python-loaded-functions-sha256-v2"


class ObservationError(Exception):
    """An observation failed; details must not reach the terminal."""


def _read_cases(path: Path) -> list[dict[str, str]]:
    # Reject a private input placed anywhere under a Git worktree, including
    # an untracked path that could later be added by accident.
    probe = subprocess.run(
        ["git", "-C", str(path.resolve().parent), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, timeout=5, check=False,
    )
    if probe.returncode == 0:
        raise ObservationError("private cases must stay outside Git")
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = raw["queries"]
    if not isinstance(cases, list) or not cases:
        raise ObservationError("invalid cases")
    for case in cases:
        if not isinstance(case, dict) or not all(
            isinstance(case.get(key), str) and case[key].strip()
            for key in ("query", "expected_node_id")
        ):
            raise ObservationError("invalid case")
    return cases


def _health(
    url: str, timeout: float, verify: ssl.SSLContext | bool | None,
    token: str | None = None,
) -> dict[str, Any]:
    # The local HTTP endpoint normally needs no TLS. For the TLS bridge, use
    # the same trust settings as the MCP client.
    context = verify if isinstance(verify, ssl.SSLContext) else None
    if verify is False:
        context = ssl._create_unverified_context()
    request = Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urlopen(request, timeout=timeout, context=context) as response:
        return json.load(response)


def _known_digest(identity: Any) -> str:
    if (
        not isinstance(identity, dict)
        or identity.get("status") != "known"
        or identity.get("scheme") != CODE_IDENTITY_SCHEME
        or not isinstance(identity.get("digest"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", identity["digest"])
    ):
        raise ObservationError("loaded-code identity unavailable")
    return identity["digest"]


def _validated_identity(
    health: dict[str, Any], prior_boot_id: str, prior_digest: str, prior_scheme: str
) -> tuple[str, str]:
    identity = health.get("code_identity") or {}
    boot_id = health.get("boot_id")
    if not health.get("ok") or not isinstance(boot_id, str) or not boot_id:
        raise ObservationError("unhealthy service")
    digest = _known_digest(identity)
    if prior_scheme != identity["scheme"]:
        raise ObservationError("prior and current loaded-code schemes differ")
    if boot_id == prior_boot_id or digest == prior_digest:
        raise ObservationError("reader has not taken up changed loaded code")
    return boot_id, digest


def _payload(result: Any) -> dict[str, Any]:
    value = result.structured_content
    if value is None:
        value = json.loads(result.content[0].text)
    if isinstance(value, dict) and set(value) == {"result"}:
        value = value["result"]
    if not isinstance(value, dict) or result.is_error:
        raise ObservationError("MCP recall failed")
    return value


async def _recall(
    url: str,
    cases: list[dict[str, str]],
    scope: str | None,
    token: str | None,
    verify: ssl.SSLContext | bool | None,
    timeout: float,
    max_results: int,
) -> list[dict[str, Any]]:
    from fastmcp import Client

    kwargs: dict[str, Any] = {"timeout": timeout, "init_timeout": timeout}
    if token:
        kwargs["auth"] = token
    if verify is not None:
        kwargs["verify"] = verify
    observations: list[dict[str, Any]] = []
    async with Client(url, **kwargs) as client:
        for case in cases:
            started = time.monotonic()
            request: dict[str, Any] = {
                "query": case["query"], "max_results": max_results,
            }
            if scope is not None:
                request["scope"] = scope
            result = _payload(await client.call_tool("memory_recall", request))
            ranked = [item["id"] for item in result["results"]]
            rank = next(
                (index for index, node_id in enumerate(ranked, 1)
                 if node_id == case["expected_node_id"]), None
            )
            observations.append({
                "found": rank is not None,
                "rank": rank,
                "returned": len(ranked),
                "response_sha256": hashlib.sha256(
                    json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "duration_ms": round((time.monotonic() - started) * 1000, 1),
            })
    return observations


def _process_identity(pid: int) -> dict[str, Any]:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        raise ObservationError("invalid process")
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid, "start_ticks": fields[19],
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "--no-replace-objects", "-C", str(root), *args], capture_output=True, check=True, timeout=20,
    ).stdout


def _product_sources(source: dict[str, bytes], anchors: list[str]) -> list[str]:
    """Inventory the server's local import closure, including lazy imports."""
    prefix = "src/living_memory/"
    modules = {
        path[len(prefix):-3]: path for path in source
        if path.startswith(prefix) and path.endswith(".py")
    }
    pending = ["server", "storage", "retrieval"]
    selected = {modules["__init__"], "pyproject.toml"}
    while pending:
        module = pending.pop()
        if module not in modules or modules[module] in selected:
            continue
        selected.add(modules[module])
        for node in ast.walk(ast.parse(source[modules[module]], filename=modules[module])):
            if isinstance(node, ast.Import):
                pending.extend(
                    name.name.removeprefix("living_memory.").split(".", 1)[0]
                    for name in node.names if name.name.startswith("living_memory.")
                )
            elif isinstance(node, ast.ImportFrom):
                if node.module == "living_memory":
                    pending.extend(name.name for name in node.names)
                elif node.module and node.module.startswith("living_memory."):
                    pending.append(node.module.split(".")[1])
                elif node.level == 1 and node.module:
                    pending.append(node.module.split(".", 1)[0])
    if not set(anchors) <= selected:
        raise ObservationError("loaded anchors are outside the consumer source inventory")
    return sorted(selected)


def _verify_commit(root: Path, commit: str) -> None:
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit):
        raise ObservationError("expected a full published commit ID")
    revision = _git(root, "rev-parse", "--verify", commit + "^{commit}").decode().strip()
    if revision != commit:
        raise ObservationError("candidate is not a commit")


def _published_sources(root: Path, commit: str) -> dict[str, bytes]:
    _verify_commit(root, commit)
    paths = _git(root, "ls-tree", "--full-tree", "-rz", "--name-only", commit, "src/living_memory")
    files = [os.fsdecode(path) for path in paths.split(b"\0") if path.endswith(b".py")]
    if "src/living_memory/server.py" not in files or "src/living_memory/__init__.py" not in files:
        raise ObservationError("candidate package unavailable")
    return {path: _git(root, "show", f"{commit}:{path}") for path in [*files, "pyproject.toml"]}


def _candidate_identity(source: dict[str, bytes], installed_root: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lm-candidate-") as temporary:
        directory = Path(temporary)
        for relative, content in source.items():
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        # Separate interpreter and temporary database reproduce the server's
        # import population without reading or changing the real memory DB.
        # server.sh adds one dependency target ahead of the interpreter's
        # packages. Leave a standalone venv's import order intact.
        deps = os.environ.get("LIVING_MEMORY_DEPS_DIR")
        if not deps:
            cache = os.path.realpath(installed_root / ".cache/python-deps")
            active_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
            if any(os.path.realpath(p) == cache for p in active_paths if p):
                deps = cache
        import_paths = [str(directory / "src")]
        if deps:
            import_paths.append(os.path.abspath(deps))
        program = """
import json
from pathlib import Path
import sys

sys.path[:0] = sys.argv[1:]
from living_memory import server

if Path(server.__file__).resolve() != Path("src/living_memory/server.py").resolve():
    raise RuntimeError("unexpected candidate origin")
instance = server.create_mcp_server(Path("candidate.sqlite3"))
print(json.dumps(server._loaded_code_identity()))
instance.memory_store.close()
"""
        environment = dict(os.environ, LIVING_MEMORY_EMBEDDING_BACKEND="hash")
        environment.pop("PYTHONPATH", None)
        environment.pop("LM_AUTH_TOKEN", None)
        result = subprocess.run(
            [sys.executable, "-B", "-c", program, *import_paths], cwd=directory, env=environment,
            capture_output=True, text=True, check=True, timeout=60,
        )
        identity = json.loads(result.stdout)
    _known_digest(identity)
    return identity


def _installed_sources(
    root: Path, source: dict[str, bytes], process: dict[str, Any],
) -> dict[str, tuple[int, int, str]]:
    boot_seconds = next(
        int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
        if line.startswith("btime ")
    )
    started_ns = boot_seconds * 1_000_000_000 + (
        int(process["start_ticks"]) * 1_000_000_000 // os.sysconf("SC_CLK_TCK")
    )
    state = {}
    for relative, expected in source.items():
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ObservationError("installed source escaped checkout")
        before = path.stat()
        actual = path.read_bytes()
        after = path.stat()
        if (
            actual != expected or before != after
            or after.st_mtime_ns >= started_ns or after.st_ctime_ns >= started_ns
        ):
            raise ObservationError("installed source differs or changed after reader start")
        state[relative] = (after.st_mtime_ns, after.st_ctime_ns, hashlib.sha256(actual).hexdigest())
    return state


def _installed_import(root: Path, process: dict[str, Any]) -> None:
    # Resolve using the consumer's interpreter, cwd and environment. This checks
    # the editable install (including its direct_url metadata when present),
    # while also permitting an explicit PYTHONPATH used by local test readers.
    proc = Path(f"/proc/{process['pid']}")
    environment = dict(
        item.decode().split("=", 1) for item in (proc / "environ").read_bytes().split(b"\0") if item
    )
    program = (
        "import importlib.util, importlib.metadata, json; "
        "spec = importlib.util.find_spec('living_memory.storage'); "
        "dist = importlib.metadata.distribution('living-memory'); "
        "print(json.dumps({'origin': spec.origin, 'direct_url': dist.read_text('direct_url.json')}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=(proc / "cwd").resolve(), env=environment,
        capture_output=True, text=True, check=True, timeout=20,
    )
    observed = json.loads(result.stdout)
    if Path(observed["origin"]).resolve() != (root / "src/living_memory/storage.py").resolve():
        raise ObservationError("consumer import does not resolve to installed source")
    direct = json.loads(observed["direct_url"] or "{}")
    if direct.get("dir_info", {}).get("editable"):
        from urllib.parse import unquote, urlparse
        location = urlparse(direct["url"])
        explicit = str((root / "src").resolve()) in environment.get("PYTHONPATH", "").split(os.pathsep)
        if not explicit and (location.scheme != "file" or Path(unquote(location.path)).resolve() != root.resolve()):
            raise ObservationError("editable install names different source")


def _native_before(args: Any, health: dict[str, Any], base: str, token: str | None,
                   verify: ssl.SSLContext | bool | None) -> dict[str, Any]:
    if not ipaddress.ip_address(args.host).is_loopback or not token:
        raise ObservationError("native observation requires authenticated loopback")
    if not args.installed_root or not args.published_commit or not args.loaded_path:
        raise ObservationError("native observation requires exact published source")
    admin = _health(base.replace("/mcp/", "/admin/info"), args.timeout, verify, token)
    if not admin.get("ok") or any(admin.get(key) != health.get(key) for key in ("boot_id", "code_identity")):
        raise ObservationError("admin and health identify different readers")
    process = _process_identity(admin["process_id"])
    if Path(f"/proc/{process['pid']}/exe").resolve() != Path(sys.executable).resolve():
        raise ObservationError("candidate and reader use different interpreters")
    _installed_import(args.installed_root, process)
    if _git(args.installed_root, "rev-parse", "HEAD").decode().strip() != args.published_commit:
        raise ObservationError("installed checkout is not the published commit")
    source = _published_sources(args.installed_root, args.published_commit)
    if len(set(args.loaded_path)) != len(args.loaded_path) or any(
        path == "pyproject.toml" or path not in source for path in args.loaded_path
    ):
        raise ObservationError("invalid loaded population")
    loaded = {
        path: hashlib.sha256(source[path]).hexdigest()
        for path in _product_sources(source, args.loaded_path)
    }
    candidate_identity = _candidate_identity(source, args.installed_root)
    if candidate_identity != health["code_identity"]:
        raise ObservationError("loaded functions differ from published candidate")
    source_state = _installed_sources(args.installed_root, source, process)
    return dict(process=process, source=source, source_state=source_state,
                loaded_bytes=loaded, candidate_digest=candidate_identity["digest"])


def _native_projection(args: Any, state: dict[str, Any], receipt: dict[str, Any],
                       base: str, token: str | None,
                       verify: ssl.SSLContext | bool | None) -> dict[str, Any]:
    admin = _health(base.replace("/mcp/", "/admin/info"), args.timeout, verify, token)
    process = _process_identity(admin["process_id"])
    if (
        process != state["process"] or admin.get("boot_id") != receipt["boot_id"]
        or admin.get("code_identity") != receipt["code_identity"]
        or _installed_sources(args.installed_root, state["source"], process) != state["source_state"]
        or _git(args.installed_root, "rev-parse", "HEAD").decode().strip() != args.published_commit
    ):
        raise ObservationError("consumer changed during native observation")
    _installed_import(args.installed_root, process)
    response_hash = hashlib.sha256(
        json.dumps([item["response_sha256"] for item in receipt["cases"]]).encode()
    ).hexdigest()
    return dict(
        receipt, pid=process["pid"], start_identity=json.dumps(process, sort_keys=True),
        observer_identity=json.dumps(_process_identity(os.getpid()), sort_keys=True),
        response_sha256=response_hash, behavior_passed=receipt["ok"],
        loaded_commit=args.published_commit, loaded_bytes=state["loaded_bytes"],
        source_binding={
            "scheme": "published-source-projection-with-loaded-functions-v1",
            "candidate_digest": state["candidate_digest"],
            "installed_import_matches": True,
            "installed_sources_match": True,
            "installed_sources_predate_process": True,
            "installed_sources_unchanged_during_observation": True,
            "limitation": "Function identity covers Python code objects, not module globals or original source bytes.",
        },
    )


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".observation-",
        delete=False,
    ) as stream:
        temp_path = Path(stream.name)
        os.chmod(temp_path, 0o600)
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    os.replace(temp_path, path)


def _connection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--tls", action="store_true")
    parser.add_argument("--ca-cert", type=Path)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--timeout", type=float, default=20.0)


def _disk_comparison(root: Path, source: dict[str, bytes]) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ObservationError("installed root unavailable")

    def raise_error(error: OSError) -> None:
        raise error

    package = root / "src/living_memory"
    if not package.resolve().is_relative_to(root):
        raise ObservationError("installed package escaped checkout")
    actual = {"pyproject.toml"} if (root / "pyproject.toml").is_file() else set()
    if package.is_dir():
        for directory, dirs, files in os.walk(package, onerror=raise_error):
            if any((Path(directory) / name).is_symlink() for name in dirs):
                raise ObservationError("installed package contains a linked directory")
            actual.update(str((Path(directory) / name).relative_to(root))
                          for name in files if name.endswith(".py"))
    different = set(source) ^ actual
    for relative in source.keys() & actual:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            different.add(relative)
        elif path.read_bytes() != source[relative]:
            different.add(relative)
    return {"status": "mismatch" if different else "match",
            "different_paths": sorted(different), "candidate_files": len(source)}


def _process_comparison(args: argparse.Namespace, source: dict[str, bytes]) -> dict[str, Any]:
    try:
        expected = _candidate_identity(source, args.installed_root)["digest"]
    except (OSError, subprocess.SubprocessError, ObservationError, ValueError):
        return {"status": "unknown", "reason": "candidate function identity unavailable"}

    base = build_url(args.host, args.port, tls=args.tls)
    try:
        health = _health(base.replace("/mcp/", "/health"), args.timeout,
                         build_verify(args.ca_cert, insecure=args.insecure))
        if not isinstance(health, dict) or health.get("ok") is not True:
            raise ObservationError("unhealthy service")
        boot_id = health.get("boot_id")
        if not isinstance(boot_id, str) or not boot_id:
            raise ObservationError("service boot identity unavailable")
        digest = _known_digest(health.get("code_identity"))
    except Exception:  # Keep raw transport diagnostics out of the public report.
        return {"status": "unknown", "candidate_digest": expected,
                "reason": "healthy service identity unavailable"}
    return {"status": "match" if digest == expected else "mismatch",
            "scheme": CODE_IDENTITY_SCHEME, "candidate_digest": expected,
            "service_digest": digest, "boot_id": boot_id}


def _contribution_comparison(root: Path, span: str, published: str) -> dict[str, Any]:
    base, _, tip = span.partition("..")
    for commit in (base, tip, published):
        _verify_commit(root, commit)
    for older, newer in ((base, tip), (tip, published)):
        try:
            _git(root, "merge-base", "--is-ancestor", older, newer)
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 1:
                raise
            return {"status": "mismatch", "reason": f"{older} is not an ancestor of {newer}"}

    changes = _git(root, "diff-tree", "-r", "-z", "--name-only", "--no-renames",
                   "--ignore-submodules=none", base, tip)
    paths = set(changes.split(b"\0")) - {b""}
    if not paths:
        return {"status": "unknown", "reason": "empty contribution"}

    def entries(commit: str) -> dict[bytes, bytes]:
        # Include trees: a deleted file replaced by a directory is no longer absent.
        result = {}
        for record in _git(root, "ls-tree", "--full-tree", "-rtz", commit).split(b"\0"):
            entry, _, path = record.partition(b"\t")
            if path in paths:
                result[path] = entry
        return result

    before, after = entries(tip), entries(published)
    different = sorted(os.fsdecode(p) for p in paths if before.get(p) != after.get(p))
    return {"status": "mismatch" if different else "match",
            "paths": sorted(map(os.fsdecode, paths)), "different_paths": different}


def _compare(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Compare candidate sources and covered functions.")
    parser.add_argument("--installed-root", required=True, type=Path)
    parser.add_argument("--candidate-root", type=Path, help="Git repository (default: installed root)")
    parser.add_argument("--published-commit", required=True, help="Full ID of a trusted candidate commit")
    parser.add_argument("--contribution", metavar="BASE..TIP",
                        help="Also verify the net changes between two full commit IDs in this candidate")
    _connection_options(parser)
    args = parser.parse_args(argv)
    root = args.candidate_root or args.installed_root
    try:
        source = _published_sources(root, args.published_commit)
    except (OSError, subprocess.SubprocessError, ObservationError):
        disk = process = {"status": "unknown", "reason": "candidate sources unavailable"}
    else:
        process = _process_comparison(args, source)
        try:
            disk = _disk_comparison(args.installed_root, source)
        except (OSError, ObservationError):
            disk = {"status": "unknown", "reason": "installed sources unavailable"}
    report = {"published_commit": args.published_commit, "disk": disk,
              "process": process, "recall": "not_run"}
    statuses = {disk["status"], process["status"]}
    if args.contribution is not None:
        try:
            comparison = _contribution_comparison(root, args.contribution, args.published_commit)
        except (OSError, subprocess.SubprocessError, ObservationError):
            comparison = {"status": "unknown", "reason": "contribution history unavailable or invalid"}
        report["contribution"] = {"range": args.contribution, **comparison}
        statuses.add(comparison["status"])
    print(json.dumps(report, sort_keys=True))
    if "mismatch" in statuses:
        return 1
    return 2 if "unknown" in statuses else 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["compare"]:
        return _compare(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-file", type=Path, required=True)
    parser.add_argument("--prior-boot-id", required=True)
    parser.add_argument("--prior-digest", required=True)
    parser.add_argument("--prior-scheme", required=True)
    parser.add_argument("--output", type=Path, required=True)
    _connection_options(parser)
    parser.add_argument("--token")
    parser.add_argument("--env-file", type=Path, default=Path.home() / ".config/living-memory/env")
    parser.add_argument("--max-results", type=int, default=6)
    parser.add_argument("--scope")
    parser.add_argument("--native-completion", action="store_true")
    parser.add_argument("--installed-root", type=Path)
    parser.add_argument("--published-commit")
    parser.add_argument("--loaded-path", action="append")
    args = parser.parse_args(argv)
    try:
        if args.max_results < 1 or args.timeout <= 0:
            raise ObservationError("invalid limit")
        cases = _read_cases(args.cases_file)
        verify = build_verify(args.ca_cert, insecure=args.insecure)
        base = build_url(args.host, args.port, tls=args.tls)
        health = _health(base.replace("/mcp/", "/health"), args.timeout, verify)
        boot_id, digest = _validated_identity(
            health, args.prior_boot_id, args.prior_digest, args.prior_scheme
        )
        token = resolve_token(args.token, env_file=args.env_file)
        native = _native_before(args, health, base, token, verify) if args.native_completion else None
        results = asyncio.run(_recall(base, cases, args.scope, token, verify, args.timeout, args.max_results))
        after = _health(base.replace("/mcp/", "/health"), args.timeout, verify)
        if (
            after.get("boot_id") != boot_id
            or after.get("code_identity") != health.get("code_identity")
        ):
            raise ObservationError("reader changed during observation")
        receipt = {
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "ok": all(item["found"] for item in results),
            "boot_id": boot_id,
            "prior_boot_id": args.prior_boot_id,
            "prior_digest": args.prior_digest,
            "code_identity": {
                "status": "known", "scheme": CODE_IDENTITY_SCHEME,
                "digest": digest,
                "git_revision": health["code_identity"].get("git_revision"),
            },
            "max_results": args.max_results,
            "unscoped_recall": args.scope is None,
            "cases": results,
        }
        if native is not None:
            receipt = _native_projection(args, native, receipt, base, token, verify)
        _write_receipt(args.output, receipt)
        if args.native_completion:
            print(json.dumps(receipt, sort_keys=True))
        else:
            print("OK: observation recorded" if receipt["ok"] else "FAIL: expected recall result absent")
        return 0 if receipt["ok"] else 1
    except ObservationError as error:
        print(f"CANNOT CHECK: {error}", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001 - do not leak private tool/transport errors
        print("CANNOT CHECK: observation unavailable", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
