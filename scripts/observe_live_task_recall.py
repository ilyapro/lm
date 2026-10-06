#!/usr/bin/env python3
"""Observe a restarted LM reader through /health and public MCP recall.

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
import io
import ipaddress
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any
from urllib.request import Request, urlopen

from check_deployed_protocol import build_url, build_verify, resolve_token


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


def _validated_identity(
    health: dict[str, Any], prior_boot_id: str, prior_digest: str
) -> tuple[str, str]:
    identity = health.get("code_identity") or {}
    boot_id = health.get("boot_id")
    digest = identity.get("digest")
    if not health.get("ok") or not isinstance(boot_id, str) or not boot_id:
        raise ObservationError("unhealthy service")
    if (
        identity.get("status") != "known"
        or identity.get("scheme") != "python-loaded-functions-sha256-v1"
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        raise ObservationError("loaded-code identity unavailable")
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
            result = _payload(await client.call_tool("memory_recall", {
                "query": case["query"], "max_results": max_results,
            }))
            ranked = [item["node"]["id"] for item in result["results"]]
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
        ["git", "-C", str(root), *args], capture_output=True, check=True, timeout=20,
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


def _candidate(
    root: Path, commit: str, loaded_paths: list[str],
) -> tuple[str, dict[str, bytes], dict[str, str]]:
    revision = _git(root, "rev-parse", "--verify", commit + "^{commit}").decode().strip()
    if revision != commit or _git(root, "rev-parse", "HEAD").decode().strip() != revision:
        raise ObservationError("installed checkout is not the published commit")
    paths = _git(root, "ls-tree", "-r", "--name-only", revision, "src/living_memory").decode().splitlines()
    source = {path: _git(root, "show", f"{revision}:{path}") for path in paths if path.endswith(".py")}
    source["pyproject.toml"] = _git(root, "show", f"{revision}:pyproject.toml")
    if not source or not loaded_paths or len(set(loaded_paths)) != len(loaded_paths):
        raise ObservationError("invalid loaded population")
    if any(path not in source for path in loaded_paths):
        raise ObservationError("loaded path is not a published Python source")
    product_paths = _product_sources(source, loaded_paths)
    archive = _git(root, "archive", revision, "src/living_memory", "pyproject.toml")
    with tempfile.TemporaryDirectory(prefix="lm-candidate-") as temporary:
        directory = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(archive)) as packed:
            packed.extractall(directory, filter="data")
        # Separate interpreter and temporary database reproduce the server's
        # import population without reading or changing the real memory DB.
        program = (
            "import json; from pathlib import Path; "
            "from living_memory.server import create_mcp_server, _loaded_code_identity; "
            "server = create_mcp_server(Path('candidate.sqlite3')); "
            "print(json.dumps(_loaded_code_identity())); server.memory_store.close()"
        )
        environment = dict(os.environ, PYTHONPATH=str(directory / "src"),
                           LIVING_MEMORY_EMBEDDING_BACKEND="hash")
        environment.pop("LM_AUTH_TOKEN", None)
        result = subprocess.run(
            [sys.executable, "-c", program], cwd=directory, env=environment,
            capture_output=True, text=True, check=True, timeout=60,
        )
        identity = json.loads(result.stdout)
    if identity.get("status") != "known":
        raise ObservationError("candidate function identity unavailable")
    return identity["digest"], source, {
        path: hashlib.sha256(source[path]).hexdigest() for path in product_paths
    }


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
    candidate_digest, source, loaded = _candidate(
        args.installed_root, args.published_commit, args.loaded_path,
    )
    if candidate_digest != health["code_identity"]["digest"]:
        raise ObservationError("loaded functions differ from published candidate")
    source_state = _installed_sources(args.installed_root, source, process)
    return dict(process=process, source=source, source_state=source_state,
                loaded_bytes=loaded, candidate_digest=candidate_digest)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-file", type=Path, required=True)
    parser.add_argument("--prior-boot-id", required=True)
    parser.add_argument("--prior-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--tls", action="store_true")
    parser.add_argument("--ca-cert", type=Path)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--token")
    parser.add_argument("--env-file", type=Path, default=Path.home() / ".config/living-memory/env")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max-results", type=int, default=6)
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
            health, args.prior_boot_id, args.prior_digest
        )
        token = resolve_token(args.token, env_file=args.env_file)
        native = _native_before(args, health, base, token, verify) if args.native_completion else None
        results = asyncio.run(_recall(base, cases, token, verify, args.timeout, args.max_results))
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
                "status": "known", "scheme": "python-loaded-functions-sha256-v1",
                "digest": digest,
                "git_revision": health["code_identity"].get("git_revision"),
            },
            "max_results": args.max_results,
            "unscoped_recall": True,
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
