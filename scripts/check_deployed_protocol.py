#!/usr/bin/env python3
"""check_deployed_protocol.py — does a deployed host serve *this* checkout's protocol?

The checker compares the effective static protocol for a selected configuration
with what MCP serves. The bounded, data-derived Memory nearby tail is reported
as host data. Text equality does not establish the loaded code revision;
``/admin/info`` and ``/health`` expose boot and code identity separately.

Expected texts come from the same effective projection used by the server's
tool registration. Local defaults read the service's env file; other targets
need an explicit expected prompt arm and default scope (or an explicit env file).

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

    0  the host serves the selected effective static protocol
    1  drift: a core text differs, a tool is missing, or configuration differs
    2  cannot check: transport, auth, or unknown expectations

See docs/deployment.md for the rollout procedure this check verifies.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import json
import os
import re
import ssl
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    value = str(candidate)
    if value not in sys.path:
        sys.path.insert(0, value)

from living_memory import server as lm_server  # noqa: E402
from living_memory import instructions_map as lm_map  # noqa: E402

# Defaults describe the local user unit (see docs/deployment.md): HTTP on
# loopback 8765, --default-scope global, bearer token in the EnvironmentFile.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_SCOPE = "global"
DEFAULT_ENV_FILE = Path.home() / ".config" / "living-memory" / "env"
DEFAULT_TIMEOUT_SECONDS = 20.0

INSTRUCTIONS_KEY = "server instructions"

PROTOCOL_TOOLS = frozenset(
    ("memory_recall", "memory_remember", "memory_teach", "memory_lookup")
)

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


def expected_texts(
    default_scope: str = DEFAULT_SCOPE,
    prompt_arm: str = lm_server.DEFAULT_EXPLICIT_FEEDBACK_PROMPT_ARM,
) -> dict[str, str]:
    """The checkout's effective static protocol for one expected configuration."""
    try:
        return lm_server._effective_protocol_texts(default_scope, prompt_arm)
    except (ValueError, AttributeError, NameError) as exc:
        raise CheckError(f"cannot project checkout protocol: {exc}") from exc


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


def resolve_prompt_arm(explicit: str | None, *, env_file: Path | None) -> str:
    """Resolve the expected arm from CLI or the target's specified env file.

    Unlike the server's fallback for an unknown env value, a checker must not
    quietly select another expected variant when the target is ambiguous.
    Neither the file nor any other settings from it appear in diagnostics.
    """
    if explicit is not None:
        return explicit
    if env_file is None:
        raise CheckError(
            "expected LM_EXPLICIT_FEEDBACK_PROMPT is unknown; pass "
            "--prompt-arm optional|mandatory or --env-file for the target"
        )
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CheckError(
            f"cannot read expected configuration from {env_file}: {exc.strerror}; "
            "pass --prompt-arm optional|mandatory"
        ) from exc
    values = [
        line.strip().split("=", 1)[1].strip().strip('"').strip("'").lower()
        for line in lines
        if line.strip().startswith("LM_EXPLICIT_FEEDBACK_PROMPT=")
    ]
    if not values or values[-1] == "":
        return lm_server.DEFAULT_EXPLICIT_FEEDBACK_PROMPT_ARM
    arm = values[-1]
    if arm not in lm_server.EXPLICIT_FEEDBACK_PROMPT_ARMS:
        raise CheckError(
            f"unsupported LM_EXPLICIT_FEEDBACK_PROMPT in {env_file}; "
            "pass --prompt-arm optional|mandatory"
        )
    return arm


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


def _memory_nearby_section(text: str) -> tuple[str, str | None]:
    """Separate only the canonical, bounded data tail from instructions.

    A malformed heading, extra line or suffix remains part of the compared
    core. This boundary must never excuse edits elsewhere in instructions.
    """
    boundary = "\n" + lm_map.HEADING + "\n"
    if text.count(boundary) != 1:
        return text, None
    core, tail = text.split(boundary, 1)
    suffix = "\n" + lm_map.CLOSING + "\n"
    if not tail.endswith(suffix):
        return text, None
    line = tail[: -len(suffix)]
    if not line.startswith(lm_map.LINE_PREFIX) or "\n" in line:
        return text, None
    section = lm_map.HEADING + "\n" + tail
    if len(section.rstrip("\n")) > lm_map.MAX_SECTION_CHARS:
        return text, None
    data = line[len(lm_map.LINE_PREFIX) :]
    if data.endswith(lm_map.MORE_MARKER):
        data = data[: -len(lm_map.MORE_MARKER)]
    items = data.split(lm_map.SEPARATOR)
    if not 1 <= len(items) <= lm_map.MAX_LABELS:
        return text, None
    for item in items:
        label, separator, count = item.rpartition("(")
        if (
            separator != "("
            or not count.endswith(")")
            or not count[:-1].isdigit()
            or not 0 < int(count[:-1]) <= lm_map.MAX_COUNT
            or not 1 <= len(label) <= lm_map.MAX_LABEL_CHARS
            or not all(char.isalpha() or char in " …" for char in label)
        ):
            return text, None
    return core, section


def compare(expected: dict[str, str], served: dict[str, str]) -> list[Difference]:
    """Every effective static text the host does not serve byte-identically."""

    differences: list[Difference] = []
    for name, expected_text in expected.items():
        served_text = served.get(name)
        if name == INSTRUCTIONS_KEY and served_text is not None:
            served_text, _ = _memory_nearby_section(served_text)
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

    The verdict stays non-zero: the explicitly expected configuration differs.
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
    served: dict[str, str],
    differences: list[Difference],
    *,
    revision: str,
    prompt_arm: str,
) -> str:
    source = Path(getattr(lm_server, "__file__", "?")).resolve()
    header = f"checkout: {source} @ {revision}"
    scope = _served_scope(expected[INSTRUCTIONS_KEY])
    settings = f"expected configuration: default scope={scope}, prompt arm={prompt_arm}"
    _, map_section = _memory_nearby_section(served.get(INSTRUCTIONS_KEY, ""))
    host_data = (
        f"host data: canonical Memory nearby section present "
        f"({len(map_section)} chars, sha256={_sha256(map_section)}); "
        "excluded from static-text comparison"
        if map_section is not None else "host data: no canonical Memory nearby section"
    )
    if not differences:
        return "\n".join(
            [
                f"OK: {url} serves this checkout's selected static protocol "
                f"({len(expected)} checked: {', '.join(sorted(expected))}).",
                header,
                settings,
                host_data,
            ]
        )
    names = ", ".join(difference.name for difference in differences)
    lines = [
        f"DRIFT: {url} differs from this checkout's selected static protocol.",
        header,
        settings,
        host_data,
        f"{len(differences)} of {len(expected)} protocol texts differ: {names}.",
        "",
    ]
    for difference in differences:
        lines.extend(_text_report(difference))
        if difference.name == "memory_recall" and difference.served is not None:
            other = next(
                arm for arm in lm_server.EXPLICIT_FEEDBACK_PROMPT_ARMS
                if arm != prompt_arm
            )
            if difference.served == expected_texts(prompt_arm=other)["memory_recall"]:
                lines.append(
                    f"    note: served recall description matches prompt arm={other}; "
                    f"this check expects {prompt_arm}."
                )
        lines.append("")
    if all(_scope_only_note(difference) for difference in differences):
        # Every difference is the host's --default-scope, not its code: saying
        # "behind master" here would send the operator after a phantom rollout.
        lines.append(
            "The static protocol text matches at the host's scope; the "
            "selected default scope differs (configuration drift)."
        )
    else:
        lines.append(
            "Measured protocol text differs. This does not establish code age; "
            "check the selected configuration and boot/code identity separately "
            "(docs/deployment.md)."
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
    prompt_arm: str,
) -> str:
    different_names = {difference.name for difference in differences}
    _, map_section = _memory_nearby_section(served.get(INSTRUCTIONS_KEY, ""))
    payload = {
        "url": url,
        "ok": not differences,
        "checkout": str(Path(getattr(lm_server, "__file__", "?")).resolve()),
        "revision": revision,
        "expected_configuration": {
            "default_scope": _served_scope(expected[INSTRUCTIONS_KEY]),
            "prompt_arm": prompt_arm,
        },
        "host_data": {
            "memory_nearby_present": map_section is not None,
            "memory_nearby_chars": len(map_section) if map_section else 0,
            "memory_nearby_sha256": _sha256(map_section),
        },
        "checked": sorted(expected),
        "texts": {
            name: {
                "equal": name not in different_names,
                "expected_chars": len(text),
                "served_chars": (
                    None if served.get(name) is None else len(
                        _memory_nearby_section(served[name])[0]
                        if name == INSTRUCTIONS_KEY else served[name]
                    )
                ),
                "expected_sha256": _sha256(text),
                "served_sha256": _sha256(
                    _memory_nearby_section(served[name])[0]
                    if name == INSTRUCTIONS_KEY and name in served
                    else served.get(name)
                ),
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
        "--prompt-arm",
        choices=lm_server.EXPLICIT_FEEDBACK_PROMPT_ARMS,
        help=(
            "expected LM_EXPLICIT_FEEDBACK_PROMPT; local default reads "
            "--env-file, explicit targets must set this or --env-file"
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
    supplied = sys.argv[1:] if argv is None else argv
    args = _build_parser().parse_args(supplied)
    url = build_url(args.host, args.port, tls=args.tls)
    try:
        local_target = (
            args.host in ("127.0.0.1", "localhost", "::1")
            and args.port == (8766 if args.tls else 8765)
        )
        explicit_env_file = any(
            arg == "--env-file" or arg.startswith("--env-file=") for arg in supplied
        )
        explicit_scope = any(
            arg == "--default-scope" or arg.startswith("--default-scope=")
            for arg in supplied
        )
        if not local_target and not explicit_scope:
            raise CheckError(
                "target default scope is unknown; pass --default-scope for this target"
            )
        target_env_file = args.env_file if local_target or explicit_env_file else None
        prompt_arm = resolve_prompt_arm(
            args.prompt_arm,
            env_file=target_env_file,
        )
        expected = expected_texts(args.default_scope, prompt_arm)
        token = resolve_token(args.token, env_file=target_env_file)
        try:
            verify = build_verify(args.ca_cert, insecure=args.insecure)
        except OSError as exc:
            raise CheckError(f"cannot configure TLS verification: {exc}") from exc
        served = fetch_served_texts(
            url,
            token=token,
            verify=verify,
            timeout=args.timeout,
        )
    except CheckError as exc:
        print(f"CANNOT CHECK: {exc}", file=sys.stderr)
        return 2
    differences = compare(expected, served)
    revision = checkout_revision()
    if args.json:
        print(render_json(
            url, expected, served, differences,
            revision=revision, prompt_arm=prompt_arm,
        ))
    else:
        print(render_report(
            url, expected, served, differences,
            revision=revision, prompt_arm=prompt_arm,
        ))
    return 1 if differences else 0


if __name__ == "__main__":
    raise SystemExit(main())
