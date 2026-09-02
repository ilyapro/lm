#!/usr/bin/env python3
"""check_deployed_protocol.py — does a deployed host serve *this* checkout's protocol?

The MCP protocol texts (the server instructions plus the four protocol-bearing
tool descriptions) are the only revision signal a running Living Memory host
exposes: ``/admin/info`` reports pid, boot_id, started_at, uptime, default scope
and argv, but no code revision. This script closes that gap. It connects to a
host over MCP, reads what the host actually serves, and compares it byte-for-byte
with the values imported from the checkout this file lives in.

The expected texts are *imported*, never copied here: ``_server_instructions``
and the authoritative tool descriptions come straight from
``living_memory.server`` of this checkout, so the check keeps working when those
texts change and never asserts a stale copy of them.

Usage::

    # local unit (living-memory.service on 127.0.0.1:8765), token from
    # ~/.config/living-memory/env
    python3 scripts/check_deployed_protocol.py

    # the TLS bridge (living-memory-tls.service) with its self-signed cert
    python3 scripts/check_deployed_protocol.py --tls --port 8766 \
        --ca-cert ~/.config/living-memory/tls/cert.pem

    # a throwaway instance started from a worktree on a spare port
    python3 scripts/check_deployed_protocol.py --port 8799 --token "$TOKEN"

Exit codes::

    0  the host serves exactly this checkout's protocol texts
    1  drift: at least one text differs, or a protocol tool is missing — when
       this checkout is master, the host is behind it
    2  the check could not be performed (connection, auth, or usage error)

See docs/deployment.md for the rollout procedure this check verifies.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import inspect
import json
import os
import re
import ssl
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import CodeType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from living_memory import server as lm_server  # noqa: E402

# Defaults describe the local user unit (see docs/deployment.md): HTTP on
# loopback 8765, --default-scope global, bearer token in the EnvironmentFile.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_SCOPE = "global"
DEFAULT_ENV_FILE = Path.home() / ".config" / "living-memory" / "env"
DEFAULT_TIMEOUT_SECONDS = 20.0

INSTRUCTIONS_KEY = "server instructions"

# tool name -> module attribute holding its description in living_memory.server.
PROTOCOL_TOOLS: dict[str, str] = {
    "memory_recall": "_RECALL_DESCRIPTION",
    "memory_remember": "_REMEMBER_DESCRIPTION",
    "memory_teach": "_TEACH_DESCRIPTION",
    "memory_lookup": "_LOOKUP_DESCRIPTION",
}

_SCOPE_LINE = re.compile(r"^Your default scope is (?P<scope>.+)\.$", re.MULTILINE)

# Sentence and clause boundaries: the diff unit for the report (display only —
# the verdict always compares the raw strings). Colons are deliberately not
# boundaries: these texts use them mid-clause ("anti-pattern: …"), and splitting
# there would strand fragments on their own lines.
_SENTENCE_SPLIT = re.compile(r"(?<=[.;!?])\s+")

_WRAP_WIDTH = 88


class CheckError(RuntimeError):
    """The comparison could not be performed at all (exit code 2)."""


@dataclass(frozen=True)
class Difference:
    """One protocol text that the host does not serve as this checkout has it."""

    name: str
    expected: str
    served: str | None  # None when the host does not expose that tool at all

    @property
    def missing(self) -> bool:
        return self.served is None


def _nested_tool_description(tool_name: str) -> str | None:
    """A FastMCP-derived description still held as a nested tool docstring.

    ``memory_lookup`` historically let FastMCP derive its description from the
    function docstring inside ``living_memory.server._register_tools``. Keep
    that server text authoritative while checkouts migrate it to the named
    ``_LOOKUP_DESCRIPTION`` constant used by the other protocol tools.
    """

    register_tools = getattr(lm_server, "_register_tools", None)
    code = getattr(register_tools, "__code__", None)
    if code is None:
        return None
    for constant in code.co_consts:
        if not isinstance(constant, CodeType) or constant.co_name != tool_name:
            continue
        raw_docstring = constant.co_consts[0] if constant.co_consts else None
        if isinstance(raw_docstring, str) and raw_docstring.strip():
            return inspect.cleandoc(raw_docstring)
    return None


def expected_texts(default_scope: str = DEFAULT_SCOPE) -> dict[str, str]:
    """The protocol texts this checkout would serve, imported from the source.

    ``default_scope`` only substitutes into the last line of the server
    instructions; it must match the ``--default-scope`` the host runs with.
    """

    texts: dict[str, str] = {
        INSTRUCTIONS_KEY: lm_server._server_instructions(default_scope)
    }
    for tool_name, attribute in PROTOCOL_TOOLS.items():
        value = getattr(lm_server, attribute, None)
        if value is None and tool_name == "memory_lookup":
            value = _nested_tool_description(tool_name)
        if not isinstance(value, str) or not value:
            raise CheckError(
                f"living_memory.server has no usable {attribute} for {tool_name}: "
                "the protocol texts moved, so this checker needs updating "
                "(scripts/check_deployed_protocol.py::PROTOCOL_TOOLS)."
            )
        texts[tool_name] = value
    return texts


def build_url(host: str, port: int, *, tls: bool) -> str:
    scheme = "https" if tls else "http"
    return f"{scheme}://{host}:{port}/mcp/"


def resolve_token(
    explicit: str | None,
    *,
    env_file: Path | None,
    environ: dict[str, str] | None = None,
) -> str | None:
    """--token, else $LM_AUTH_TOKEN, else LM_AUTH_TOKEN= in the env file.

    Returns None when no token is configured anywhere, so an unauthenticated
    instance (a worktree server started without a token) can be checked too.
    The value is never printed by this script.
    """

    if explicit:
        return explicit.strip() or None
    env = os.environ if environ is None else environ
    from_env = (env.get("LM_AUTH_TOKEN") or "").strip()
    if from_env:
        return from_env
    if env_file is None:
        return None
    try:
        raw = env_file.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("LM_AUTH_TOKEN="):
            continue
        value = line.split("=", 1)[1].strip().strip('"').strip("'")
        if value:
            return value
    return None


def build_verify(
    ca_cert: Path | None, *, insecure: bool
) -> ssl.SSLContext | bool | None:
    """TLS verification argument for the MCP client."""

    if insecure:
        return False
    if ca_cert is not None:
        return ssl.create_default_context(cafile=str(ca_cert))
    return None


async def _fetch_async(
    url: str,
    *,
    token: str | None,
    verify: ssl.SSLContext | bool | None,
    timeout: float,
) -> dict[str, str]:
    try:
        from fastmcp import Client
    except ImportError as exc:  # pragma: no cover - environment without fastmcp
        raise CheckError(
            f"fastmcp is required to talk to a deployed host: {exc}"
        ) from exc

    kwargs: dict[str, Any] = {"timeout": timeout, "init_timeout": timeout}
    if token:
        kwargs["auth"] = token
    if verify is not None:
        kwargs["verify"] = verify
    client = Client(url, **kwargs)
    async with client:  # __aenter__ performs the MCP initialize handshake.
        initialize_result = client.initialize_result
        served: dict[str, str] = {
            INSTRUCTIONS_KEY: getattr(initialize_result, "instructions", None) or ""
        }
        for tool in await client.list_tools():
            if tool.name in PROTOCOL_TOOLS:
                served[tool.name] = tool.description or ""
    return served


def fetch_served_texts(
    url: str,
    *,
    token: str | None = None,
    verify: ssl.SSLContext | bool | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, str]:
    """Server instructions + protocol tool descriptions as the host serves them."""

    try:
        return asyncio.run(
            _fetch_async(url, token=token, verify=verify, timeout=timeout)
        )
    except CheckError:
        raise
    except Exception as exc:  # noqa: BLE001 - any transport failure is "cannot check"
        hint = ""
        if not token:
            hint = (
                " (no bearer token was resolved; pass --token or --env-file if the "
                "host requires one)"
            )
        raise CheckError(
            f"could not read the protocol from {url}: {exc!r}{hint}"
        ) from exc


def compare(expected: dict[str, str], served: dict[str, str]) -> list[Difference]:
    """Every expected text the host does not serve byte-identically."""

    differences: list[Difference] = []
    for name, expected_text in expected.items():
        served_text = served.get(name)
        if served_text != expected_text:
            differences.append(
                Difference(name=name, expected=expected_text, served=served_text)
            )
    return differences


def _display_lines(text: str) -> list[str]:
    """Split a protocol text into stable diff units: one sentence per line.

    The texts are single very long lines. Wrapping them to a width would re-flow
    every line after the first edit, so a one-sentence change would read as a
    total rewrite; splitting on sentence boundaries first keeps an inserted or
    removed sentence to a single +/- line.
    """

    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        for sentence in _SENTENCE_SPLIT.split(paragraph.strip()):
            if not sentence:
                continue
            lines.extend(
                textwrap.wrap(
                    sentence,
                    width=_WRAP_WIDTH,
                    subsequent_indent="  ",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                or [sentence]
            )
    return lines


def _served_scope(text: str) -> str | None:
    match = _SCOPE_LINE.search(text)
    return match.group("scope") if match else None


def _scope_only_note(difference: Difference) -> str | None:
    """Explain an instructions mismatch caused by the host's --default-scope.

    A host serving a different default scope is a configuration difference, not
    stale code; saying so keeps the operator from chasing a rollout that already
    happened. The verdict stays non-zero — the texts really do differ.
    """

    if difference.name != INSTRUCTIONS_KEY or difference.served is None:
        return None
    host_scope = _served_scope(difference.served)
    expected_scope = _served_scope(difference.expected)
    if host_scope is None or host_scope == expected_scope:
        return None
    if lm_server._server_instructions(host_scope) != difference.served:
        return None
    return (
        f"    note: identical apart from the default scope — the host runs "
        f"--default-scope {host_scope}, this check expected {expected_scope}. "
        f"Re-run with --default-scope {host_scope} to compare the protocol itself."
    )


def _text_report(difference: Difference) -> list[str]:
    lines = [f"--- {difference.name} ---"]
    if difference.missing:
        lines.append(
            "    the host does not expose this tool at all "
            f"(expected {len(difference.expected)} chars)"
        )
        return lines
    served = difference.served or ""
    lines.append(
        f"    expected {len(difference.expected)} chars (this checkout) "
        f"vs served {len(served)} chars (host)"
    )
    note = _scope_only_note(difference)
    if note:
        lines.append(note)
    expected_lines = _display_lines(difference.expected)
    served_lines = _display_lines(served)
    diff = list(
        difflib.unified_diff(
            expected_lines,
            served_lines,
            fromfile="expected (this checkout)",
            tofile="served (host)",
            lineterm="",
            n=1,
        )
    )
    if not diff:
        lines.append(
            "    the texts differ only in whitespace (identical once wrapped)"
        )
        return lines
    lines.extend(f"    {line}" for line in diff)
    return lines


def checkout_revision() -> str:
    """Revision of the checkout the expected texts came from (best effort).

    ``--dirty`` matters here: comparing a host against uncommitted local edits
    is a different claim from comparing it against a commit, and the report
    should not hide which one the operator is looking at.
    """

    try:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else "unknown"


def render_report(
    url: str,
    expected: dict[str, str],
    differences: list[Difference],
    *,
    revision: str,
) -> str:
    source = Path(getattr(lm_server, "__file__", "?")).resolve()
    header = f"checkout: {source} @ {revision}"
    if not differences:
        return "\n".join(
            [
                f"OK: {url} serves this checkout's protocol texts "
                f"({len(expected)} checked: {', '.join(sorted(expected))}).",
                header,
            ]
        )
    names = ", ".join(difference.name for difference in differences)
    lines = [
        f"DRIFT: {url} does not serve this checkout's protocol texts.",
        header,
        f"{len(differences)} of {len(expected)} protocol texts differ: {names}.",
        "",
    ]
    for difference in differences:
        lines.extend(_text_report(difference))
        lines.append("")
    if all(_scope_only_note(difference) for difference in differences):
        # Every difference is the host's --default-scope, not its code: saying
        # "behind master" here would send the operator after a phantom rollout.
        lines.append(
            "The protocol itself matches; only the host's default scope differs, "
            "which is configuration, not stale code."
        )
    else:
        lines.append(
            "When this checkout is at master, the host is running older code: "
            "update and restart it (docs/deployment.md)."
        )
    return "\n".join(lines)


def _sha256(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_json(
    url: str,
    expected: dict[str, str],
    served: dict[str, str],
    differences: list[Difference],
    *,
    revision: str,
) -> str:
    payload = {
        "url": url,
        "ok": not differences,
        "checkout": str(Path(getattr(lm_server, "__file__", "?")).resolve()),
        "revision": revision,
        "checked": sorted(expected),
        "texts": {
            name: {
                "equal": served.get(name) == text,
                "expected_chars": len(text),
                "served_chars": (
                    None if served.get(name) is None else len(served[name])
                ),
                "expected_sha256": _sha256(text),
                "served_sha256": _sha256(served.get(name)),
            }
            for name, text in sorted(expected.items())
        },
        "differences": [difference.name for difference in differences],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the protocol texts a deployed Living Memory host serves "
            "with the ones in this checkout."
        )
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="default: %(default)s")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="default: %(default)s"
    )
    parser.add_argument(
        "--tls", action="store_true", help="connect over https (the TLS bridge)"
    )
    parser.add_argument(
        "--ca-cert",
        type=Path,
        default=None,
        help="PEM certificate to trust for --tls (a self-signed bridge cert)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="skip TLS certificate verification",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="bearer token; default: $LM_AUTH_TOKEN, then --env-file",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="EnvironmentFile to read LM_AUTH_TOKEN from (default: %(default)s)",
    )
    parser.add_argument(
        "--default-scope",
        default=DEFAULT_SCOPE,
        help=(
            "the --default-scope the host runs with; it appears in the server "
            "instructions (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="connect/read timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print a machine-readable verdict"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    url = build_url(args.host, args.port, tls=args.tls)
    try:
        expected = expected_texts(args.default_scope)
        token = resolve_token(args.token, env_file=args.env_file)
        served = fetch_served_texts(
            url,
            token=token,
            verify=build_verify(args.ca_cert, insecure=args.insecure),
            timeout=args.timeout,
        )
    except CheckError as exc:
        print(f"CANNOT CHECK: {exc}", file=sys.stderr)
        return 2
    differences = compare(expected, served)
    revision = checkout_revision()
    if args.json:
        print(render_json(url, expected, served, differences, revision=revision))
    else:
        print(render_report(url, expected, differences, revision=revision))
    return 1 if differences else 0


if __name__ == "__main__":
    raise SystemExit(main())
