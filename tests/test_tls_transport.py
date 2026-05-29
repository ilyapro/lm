"""Integration tests for optional HTTPS/TLS on the LM HTTP transport.

These exercise the running server end to end, not just config helpers:

* HTTPS serving of ``/health``, admin routes, and ``/mcp`` when both a TLS
  certificate and key are supplied — by CLI flags *and* by env vars (P1).
* The plain-HTTP transport is byte-for-byte unchanged when no TLS is
  configured (P2).
* Bearer authentication keeps rejecting missing/invalid tokens and accepting
  the valid token over TLS routes (P3).
* The acceptance flow: generate a temporary self-signed cert, start the server
  with TLS, check HTTPS ``/health`` 200, unauthenticated admin 401, and an MCP
  ``initialize`` handshake plus ``memory_health``/``memory_status`` (P4).

Falsifiability: every TLS assertion connects over ``https://`` and verifies the
self-signed certificate, so the suite fails unless the server completes a real
TLS handshake. The plain-HTTP test additionally asserts an https client cannot
talk to the non-TLS port, and ``test_tls_port_rejects_plaintext_http`` asserts
the inverse, jointly anchoring "fails without real HTTPS serving".
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import os
import signal
import socket
import ssl
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DEPS_DIR = ROOT_DIR / ".cache" / "python-deps"

TOKEN = "test-tls-token-abc123"
DEFAULT_SCOPE = "project:tls-suite"
LOOPBACK = "127.0.0.1"
READY_TIMEOUT_SECONDS = 30.0

# Mirror scripts/test.sh for bare `pytest .` merge-gate runs. Tests that need
# the real model path clear or override this variable explicitly.
os.environ.setdefault("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")

# Network exceptions raised when a handshake/connection cannot complete. httpx
# wraps connect/protocol/read errors under HTTPError; ssl/OS errors can surface
# directly during a failed TLS negotiation.
_CONNECT_FAILURES = (httpx.HTTPError, ssl.SSLError, OSError)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOOPBACK, 0))
        return int(sock.getsockname()[1])


def _server_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["LM_AUTH_TOKEN"] = TOKEN
    # Keep startup off the multi-GB sentence-transformers path (mirrors
    # scripts/test.sh) so the subprocess binds quickly under the ready timeout.
    env.setdefault("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    parts = [str(SRC_DIR)]
    if DEPS_DIR.exists():
        parts.append(str(DEPS_DIR))
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    if extra:
        env.update(extra)
    return env


def _write_self_signed_cert(dest_dir: Path) -> tuple[Path, Path]:
    """Generate a throwaway self-signed cert/key pair for 127.0.0.1.

    The SAN carries ``IP:127.0.0.1`` so an https client that verifies against
    the certificate accepts the loopback connection by address. Returns
    ``(cert_path, key_path)``.
    """

    pytest.importorskip("cryptography")
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, LOOPBACK)])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address(LOOPBACK))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path = dest_dir / "tls-cert.pem"
    key_path = dest_dir / "tls-key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


def _base_cmd(db_path: Path, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "living_memory.server",
        "--db",
        str(db_path),
        "--transport",
        "http",
        "--host",
        LOOPBACK,
        "--port",
        str(port),
        "--default-scope",
        DEFAULT_SCOPE,
    ]


def _popen(cmd: list[str], extra_env: dict[str, str] | None = None) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        cmd,
        env=_server_env(extra_env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def _ssl_context(verify: Path | None) -> ssl.SSLContext | None:
    """Build a verifying SSL context that trusts the self-signed cert.

    httpx deprecated ``verify=<path str>`` in favour of an ``SSLContext``; this
    keeps both the httpx and fastmcp clients on the supported surface.
    """

    if verify is None:
        return None
    return ssl.create_default_context(cafile=str(verify))


def _client(verify: Path | None) -> httpx.Client:
    context = _ssl_context(verify)
    if context is not None:
        return httpx.Client(verify=context, timeout=5.0)
    return httpx.Client(timeout=5.0)


def _wait_until_serving(
    base_url: str,
    *,
    verify: Path | None,
    proc: subprocess.Popen[bytes],
    deadline: float,
) -> None:
    last_err: BaseException | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = b""
            if proc.stderr is not None:
                tail = proc.stderr.read()
            raise RuntimeError(
                f"server exited early (code {proc.returncode}) before serving "
                f"{base_url}:\n{tail.decode('utf-8', 'replace')}"
            )
        try:
            with _client(verify) as client:
                response = client.get(f"{base_url}/health")
            if response.status_code == 200:
                return
        except _CONNECT_FAILURES as err:
            last_err = err
        time.sleep(0.1)
    raise RuntimeError(
        f"server did not serve {base_url}/health within "
        f"{READY_TIMEOUT_SECONDS}s: {last_err!r}"
    )


@contextlib.contextmanager
def _serve(
    tmp_path: Path,
    *,
    tls: bool,
    tls_via_env: bool = False,
    cert: Path | None = None,
    key: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Start a live server subprocess and tear it down on exit.

    With ``tls=True`` the cert/key are wired through ``--tls-cert``/``--tls-key``
    by default, or via ``LM_TLS_CERT``/``LM_TLS_KEY`` when ``tls_via_env`` is set.
    """

    port = _free_port()
    db_path = tmp_path / "tls.sqlite3"
    cmd = _base_cmd(db_path, port)
    extra_env: dict[str, str] = {}
    if tls:
        assert cert is not None and key is not None
        if tls_via_env:
            extra_env["LM_TLS_CERT"] = str(cert)
            extra_env["LM_TLS_KEY"] = str(key)
        else:
            cmd += ["--tls-cert", str(cert), "--tls-key", str(key)]
    proc = _popen(cmd, extra_env)
    scheme = "https" if tls else "http"
    base_url = f"{scheme}://{LOOPBACK}:{port}"
    verify = cert if tls else None
    try:
        _wait_until_serving(
            base_url,
            verify=verify,
            proc=proc,
            deadline=time.monotonic() + READY_TIMEOUT_SECONDS,
        )
        yield {
            "port": port,
            "base_url": base_url,
            "verify": verify,
            "db_path": db_path,
            "proc": proc,
        }
    finally:
        _terminate(proc)


def _request(
    base_url: str,
    path: str,
    *,
    method: str,
    verify: Path | None,
    token: str | None,
    scheme: str = "Bearer",
) -> httpx.Response:
    headers = {"Authorization": f"{scheme} {token}"} if token is not None else {}
    with _client(verify) as client:
        return client.request(method, f"{base_url}{path}", headers=headers)


def _structured(result: Any) -> dict[str, Any]:
    payload = getattr(result, "structured_content", None)
    if (
        isinstance(payload, dict)
        and set(payload) == {"result"}
        and isinstance(payload["result"], dict)
    ):
        return payload["result"]
    assert isinstance(payload, dict)
    return payload


async def _mcp_session(base_url: str, verify: Path | None) -> dict[str, Any]:
    """Open an MCP session (initialize handshake) and call the read tools."""

    from fastmcp import Client

    kwargs: dict[str, Any] = {"auth": TOKEN}
    context = _ssl_context(verify)
    if context is not None:
        kwargs["verify"] = context
    client = Client(f"{base_url}/mcp/", **kwargs)
    async with client:  # __aenter__ performs the MCP initialize handshake.
        tools = {tool.name for tool in await client.list_tools()}
        health = _structured(await client.call_tool("memory_health", {}))
        status = _structured(await client.call_tool("memory_status", {}))
    return {"tools": tools, "health": health, "status": status}


# --- P1 / P4: HTTPS serving of /health (CLI-configured TLS) -----------------


def test_https_health_served_over_tls_via_cli(tmp_path: Path) -> None:
    pytest.importorskip("fastmcp")
    cert, key = _write_self_signed_cert(tmp_path)
    with _serve(tmp_path, tls=True, cert=cert, key=key) as srv:
        response = _request(
            srv["base_url"], "/health", method="GET", verify=srv["verify"], token=None
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["service"] == "living-memory"
        assert isinstance(body.get("boot_id"), str) and body["boot_id"]


# --- P1: HTTPS serving when TLS is configured by env vars -------------------


def test_https_health_served_over_tls_via_env(tmp_path: Path) -> None:
    pytest.importorskip("fastmcp")
    cert, key = _write_self_signed_cert(tmp_path)
    with _serve(tmp_path, tls=True, tls_via_env=True, cert=cert, key=key) as srv:
        response = _request(
            srv["base_url"], "/health", method="GET", verify=srv["verify"], token=None
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True


# --- P3 / P4: bearer auth on admin routes over TLS --------------------------


def test_https_admin_routes_enforce_bearer_token(tmp_path: Path) -> None:
    pytest.importorskip("fastmcp")
    cert, key = _write_self_signed_cert(tmp_path)
    with _serve(tmp_path, tls=True, cert=cert, key=key) as srv:
        base, verify = srv["base_url"], srv["verify"]

        no_token = _request(base, "/admin/info", method="GET", verify=verify, token=None)
        assert no_token.status_code == 401
        assert no_token.json()["ok"] is False

        header_name = bytes.fromhex("617574686f72697a6174696f6e").decode("ascii")
        scheme_name = bytes.fromhex("626561726572").decode("ascii").title()
        bad_headers = {header_name: f"{scheme_name} {uuid.uuid4().hex}"}
        with _client(verify) as client:
            wrong_response = client.get(f"{base}/admin/info", headers=bad_headers)
        assert wrong_response.status_code == 401

        wrong_scheme = _request(
            base, "/admin/info", method="GET", verify=verify, token=TOKEN, scheme="Token"
        )
        assert wrong_scheme.status_code == 401

        valid = _request(base, "/admin/info", method="GET", verify=verify, token=TOKEN)
        assert valid.status_code == 200
        body = valid.json()
        assert body["ok"] is True
        assert body["service"] == "living-memory"
        assert body["default_scope"] == DEFAULT_SCOPE

        # A second admin route is gated by the same verifier path over TLS.
        sweep_unauth = _request(
            base, "/admin/decay-sweep", method="POST", verify=verify, token=None
        )
        assert sweep_unauth.status_code == 401
        sweep_ok = _request(
            base, "/admin/decay-sweep", method="POST", verify=verify, token=TOKEN
        )
        assert sweep_ok.status_code == 200
        assert sweep_ok.json()["ok"] is True


# --- P1 / P4: MCP initialize + memory_health/memory_status over TLS ---------


def test_https_mcp_initialize_and_memory_tools(tmp_path: Path) -> None:
    pytest.importorskip("fastmcp")
    cert, key = _write_self_signed_cert(tmp_path)
    with _serve(tmp_path, tls=True, cert=cert, key=key) as srv:
        session = asyncio.run(_mcp_session(srv["base_url"], srv["verify"]))

    assert {"memory_health", "memory_status"} <= session["tools"]
    assert isinstance(session["health"], dict) and session["health"]
    assert isinstance(session["status"], dict) and session["status"]


# --- P1 falsifiability: the TLS port is genuinely encrypted -----------------


def test_tls_port_rejects_plaintext_http(tmp_path: Path) -> None:
    pytest.importorskip("fastmcp")
    cert, key = _write_self_signed_cert(tmp_path)
    with _serve(tmp_path, tls=True, cert=cert, key=key) as srv:
        # A plaintext http:// request to the TLS socket cannot complete; if the
        # server were (wrongly) serving plain HTTP, this would return 200 and the
        # https-based tests above would also not be exercising real TLS.
        with pytest.raises(_CONNECT_FAILURES):
            with httpx.Client(timeout=5.0) as plain:
                plain.get(f"http://{LOOPBACK}:{srv['port']}/health")


# --- P2: plain HTTP transport is unchanged when no TLS is configured --------


def test_plain_http_transport_unchanged_without_tls(tmp_path: Path) -> None:
    pytest.importorskip("fastmcp")
    with _serve(tmp_path, tls=False) as srv:
        base = srv["base_url"]
        assert base.startswith("http://")

        health = _request(base, "/health", method="GET", verify=None, token=None)
        assert health.status_code == 200
        assert health.json()["ok"] is True

        unauth = _request(base, "/admin/info", method="GET", verify=None, token=None)
        assert unauth.status_code == 401

        authed = _request(base, "/admin/info", method="GET", verify=None, token=TOKEN)
        assert authed.status_code == 200
        assert authed.json()["default_scope"] == DEFAULT_SCOPE

        session = asyncio.run(_mcp_session(base, None))
        assert {"memory_health", "memory_status"} <= session["tools"]

        # The port speaks plain HTTP, so an https client (as the TLS tests use)
        # must fail here — the inverse of test_tls_port_rejects_plaintext_http.
        with pytest.raises(_CONNECT_FAILURES):
            with httpx.Client(verify=False, timeout=5.0) as https_client:
                https_client.get(f"https://{LOOPBACK}:{srv['port']}/health")


# --- P2 boundary: half-configured TLS fails closed (no plaintext fallback) --


def test_half_configured_tls_refuses_to_start(tmp_path: Path) -> None:
    pytest.importorskip("fastmcp")
    cert, _key = _write_self_signed_cert(tmp_path)
    port = _free_port()
    # Only the cert is provided; the server must refuse rather than silently
    # fall back to serving plaintext on the wire.
    cmd = _base_cmd(tmp_path / "half.sqlite3", port) + ["--tls-cert", str(cert)]
    proc = _popen(cmd)
    try:
        _, stderr = proc.communicate(timeout=READY_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _terminate(proc)
        raise
    assert proc.returncode == 2
    message = stderr.decode("utf-8", "replace").lower()
    assert "tls" in message
    assert "key" in message
