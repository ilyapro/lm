#!/usr/bin/env python3
"""Runner for the preregistered optional-vs-mandatory explicit-feedback experiment.

Follows ``artifacts/explicit-feedback/experiment/preregistration.md`` (the
"prereg"); section numbers below refer to it. Every step is a subcommand so
the whole experiment can be replayed from the prereg:

    verify-base   P1: prereg + tasks unchanged since 5ae6873, result.md absent
    snapshot      P2: backup_database of sfx (local) and alt (over ssh), mode=ro
    prepare       P2: <store>.base.sqlite3 = migrated copy with the class-B ablations
    preflight     P3: per-arm tools/list payload, smoke calls, codex-usable rule
    run           P4: the paired launch queue under the stopping rules (§9)
    leakcheck     P5/X1: read-only scan of both live stores for sandbox queries
    analyze       P5: §6 metrics, §7 falsifiers and decision table -> summary.json

Scratch state (snapshots, per-run store copies, raw agent streams) lives in
``$EFX_EXP_DIR`` (default ``~/.cache/efx_exp``), outside the repo. Per-run
exports without secrets go to ``artifacts/explicit-feedback/experiment/results``.

Live stores are only ever opened ``file:...?mode=ro``; no process started here
talks to port 8765. Sandbox servers listen on 18800-18999 with a fresh token
each, and agents are started under ``env -i`` with a strict MCP config that
names only the sandbox.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import secrets
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

WT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WT / "src"))
sys.path.insert(0, str(WT / "scripts"))

EXP_REL = Path("artifacts/explicit-feedback/experiment")
PREREG = WT / EXP_REL / "preregistration.md"
TASKS = WT / EXP_REL / "tasks.jsonl"
RESULTS = WT / EXP_REL / "results"
PREREG_COMMIT = "5ae6873"
TASKS_SHA = "1a6dfde2a5dd7765e697735f7b99c6e6a25406068253b2fea979a02a844f832f"
EXP = Path(os.environ.get("EFX_EXP_DIR", str(Path.home() / ".cache" / "efx_exp")))

LIVE = {
    "sfx": Path("/home/sfx/.local/share/living-memory/global.sqlite3"),
    "alt": Path("/home/user/.local/share/living-memory/global.sqlite3"),  # on host `alt`
}
ALT_PYTHON = "~/.local/share/lm-venv/bin/python"
ENV_FILE = "~/.config/living-memory/env"
KNOB_KEYS = (
    "LM_DEFAULT_SCOPE",
    "LM_AUTO_CONSOLIDATE_POLICY",
    "LM_RETRIEVAL_TUNING_POLICY",
    "LM_DRAIN_NEAR_DUP_SUPERSEDES",
    "LM_RECALL_NEAR_DUP_COSINE",
    "LM_MAP_POOL_COLD_QUOTA_GATE",
    "LM_MAP_POOL_COLD_SLOTS",
    "LM_MAP_CURTAIL_DECAY",
)
ARMS = ("optional", "mandatory")
AGENT_TOOLS = ("memory_recall", "memory_remember", "memory_teach", "memory_lookup")
CELL_ORDER = (("sfx", "claude"), ("alt", "claude"), ("sfx", "codex"), ("alt", "codex"))
RUN_TIMEOUT = 420
READY_TIMEOUT = 120
MAX_PAIRS = 3
NO_LAUNCH_AFTER = 105 * 60
KILL_AFTER = 120 * 60
TOKEN_BUDGET = 40_000_000
OUTPUT_BUDGET = 2_000_000
CLAUDE_MODEL = "claude-opus-5-5"
MIN_CONTAINMENT = 0.22
PERM_ROUNDS = 10_000
SEED = int(TASKS_SHA[:8], 16)
SMOKE_PROMPT = "Call memory_recall with query 'sandbox preflight' and report how many results you got."
PROVIDER_ERR = re.compile(r"rate.?limit|429|overloaded|5\d\d |unauthori[sz]ed|authentication|quota|usage limit", re.I)


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def load_tasks() -> list[dict[str, Any]]:
    return [json.loads(line) for line in TASKS.read_text().splitlines() if line.strip()]


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(WT), *args], capture_output=True, text=True, check=True).stdout.strip()


# ----------------------------------------------------------------------- P1/P2


def cmd_verify_base(_: argparse.Namespace) -> int:
    head = git("rev-parse", "HEAD")
    changed = git("diff", "--name-only", PREREG_COMMIT, "HEAD", "--", str(PREREG.relative_to(WT)), str(TASKS.relative_to(WT)))
    dirty = git("status", "--porcelain", "--", str(EXP_REL))
    out = {
        "head": head,
        "prereg_commit": git("rev-parse", PREREG_COMMIT),
        "prereg_or_tasks_changed_since_prereg": bool(changed),
        "tasks_sha256": sha256_file(TASKS),
        "tasks_sha_matches": sha256_file(TASKS) == TASKS_SHA,
        "result_md_absent": not (WT / EXP_REL / "result.md").exists(),
        "experiment_dir_dirty": dirty,
        "checked_at": now(),
    }
    dump_json(EXP / "base_check.json", out)
    print(json.dumps(out, indent=2))
    return 0 if (not changed and out["tasks_sha_matches"] and out["result_md_absent"]) else 1


def read_knobs(store: str) -> dict[str, str]:
    """Only the retrieval knobs; token and DB path never leave the env file."""

    pattern = "^(" + "|".join(KNOB_KEYS) + ")="
    if store == "sfx":
        text = subprocess.run(["grep", "-E", pattern, os.path.expanduser(ENV_FILE)], capture_output=True, text=True).stdout
    else:
        text = subprocess.run(
            ["timeout", "30", "ssh", "-o", "BatchMode=yes", "alt", f"grep -E '{pattern}' {ENV_FILE}"],
            capture_output=True, text=True,
        ).stdout
    knobs = {}
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key in KNOB_KEYS:
            knobs[key] = value.strip()
    return knobs


def live_recall_count(store: str) -> dict[str, Any]:
    sql = "select count(*), max(created_at) from recall_events"
    if store == "sfx":
        conn = sqlite3.connect(f"file:{LIVE['sfx']}?mode=ro", uri=True)
        try:
            count, last = conn.execute(sql).fetchone()
        finally:
            conn.close()
    else:
        code = (
            "import sqlite3,json;c=sqlite3.connect('file:%s?mode=ro',uri=True);"
            "print(json.dumps(c.execute('%s').fetchone()))" % (LIVE["alt"], sql)
        )
        out = subprocess.run(["timeout", "60", "ssh", "alt", f"{ALT_PYTHON} -c \"{code}\""], capture_output=True, text=True).stdout
        count, last = json.loads(out)
    return {"at": now(), "recall_events": count, "max_created_at": last}


def cmd_snapshot(_: argparse.Namespace) -> int:
    from living_memory.retrieval_harness import backup_database

    (EXP / "snap").mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {"started_at": now()}
    backup_database(LIVE["sfx"], EXP / "snap" / "sfx.raw.sqlite3")
    remote = (
        f"mkdir -p ~/efx_exp && {ALT_PYTHON} -c \"from living_memory.retrieval_harness import backup_database;"
        f"backup_database('{LIVE['alt']}', '/home/user/efx_exp/alt.raw.sqlite3')\""
    )
    alt = subprocess.run(["timeout", "1500", "ssh", "alt", remote], capture_output=True, text=True)
    if alt.returncode == 0:
        subprocess.run(["timeout", "1500", "scp", "alt:efx_exp/alt.raw.sqlite3", str(EXP / "snap" / "alt.raw.sqlite3")], check=False)
    info["alt_error"] = None if alt.returncode == 0 else alt.stderr[-500:]
    for store in ("sfx", "alt"):
        path = EXP / "snap" / f"{store}.raw.sqlite3"
        info[f"{store}_raw_sha256"] = sha256_file(path) if path.exists() else None
    info["finished_at"] = now()
    dump_json(EXP / "snapshot.json", info)
    print(json.dumps(info, indent=2))
    return 0


def cmd_prepare(_: argparse.Namespace) -> int:
    from living_memory.storage import MemoryStore

    tasks = load_tasks()
    info = load_json(EXP / "snapshot.json", {})
    for store in ("sfx", "alt"):
        raw = EXP / "snap" / f"{store}.raw.sqlite3"
        if not raw.exists():
            info[f"{store}_base"] = "not run: no snapshot"
            continue
        info.setdefault(f"{store}_raw_sha256", sha256_file(raw))
        base = EXP / "snap" / f"{store}.base.sqlite3"
        shutil.copyfile(raw, base)
        memory = MemoryStore(base)
        ablate = sorted({n for t in tasks if t["store"] == store for n in t["ablate_node_ids"]})
        done, missing = [], []
        for node_id in ablate:
            try:
                memory.soft_delete_node(node_id, "efx-ablation")
                done.append(node_id)
            except KeyError:
                missing.append(node_id)
        memory.close() if hasattr(memory, "close") else None
        del memory
        conn = sqlite3.connect(base)
        # A token rotated on the live host lives in kv and would override the
        # sandbox's fresh LM_AUTH_TOKEN (server.py: kv wins over env).
        had_kv_token = conn.execute("select count(*) from kv where key = 'auth_token'").fetchone()[0]
        conn.execute("delete from kv where key = 'auth_token'")
        conn.commit()
        conn.execute("pragma wal_checkpoint(TRUNCATE)")
        conn.execute("pragma journal_mode=DELETE")
        conn.close()
        gold = {n for t in tasks if t["store"] == store for n in t["gold_node_ids"]}
        conn = sqlite3.connect(f"file:{base}?mode=ro", uri=True)
        gold_bad = sorted(
            n for n in gold
            if (row := conn.execute("select decayed from nodes where id = ?", (n,)).fetchone()) is None or row[0]
        )
        conn.close()
        info[f"{store}_base"] = {
            "ablated": len(done),
            "ablate_missing": missing,
            "kv_auth_token_removed": bool(had_kv_token),
            "gold_missing_or_decayed": gold_bad,
            "sha256": sha256_file(base),
            "prepared_at": now(),
        }
    info["knobs"] = {store: read_knobs(store) for store in ("sfx", "alt")}
    dump_json(EXP / "snapshot.json", info)
    print(json.dumps(info, indent=2))
    return 0


# ---------------------------------------------------------------------- server

_PORT_LOCK = threading.Lock()
_PORTS_USED: set[int] = set()


def free_port() -> int:
    with _PORT_LOCK:
        for port in range(18800, 19000):
            if port in _PORTS_USED:
                continue
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    continue
            _PORTS_USED.add(port)
            return port
    raise RuntimeError("no free port in 18800-18999")


async def _list_tools(url: str, token: str) -> list[dict[str, Any]]:
    from fastmcp import Client

    async with Client(url, auth=token, timeout=20, init_timeout=20) as client:
        tools = await client.list_tools()
    return [
        {"name": t.name, "description": t.description or "", "inputSchema": t.inputSchema}
        for t in tools
    ]


def payload_stats(tools: list[dict[str, Any]]) -> dict[str, Any]:
    canonical = json.dumps(sorted(tools, key=lambda t: t["name"]), sort_keys=True, ensure_ascii=False)
    visible = [t for t in tools if t["name"] in AGENT_TOOLS]
    visible_json = json.dumps(sorted(visible, key=lambda t: t["name"]), sort_keys=True, ensure_ascii=False)
    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        tokens, method = len(enc.encode(visible_json)), "tiktoken o200k_base"
    except Exception:
        tokens, method = round(len(visible_json) / 4), "chars/4 (tiktoken not installed)"
    return {
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "tool_names": sorted(t["name"] for t in tools),
        "description_chars": {t["name"]: len(t["description"]) for t in tools},
        "schema_chars": {t["name"]: len(json.dumps(t["inputSchema"], sort_keys=True)) for t in tools},
        "visible_payload_chars": len(visible_json),
        "visible_payload_tokens": tokens,
        "token_method": method,
    }


class Sandbox:
    def __init__(self, store: str, arm: str, run_dir: Path, knobs: dict[str, str]):
        self.store, self.arm, self.run_dir, self.knobs = store, arm, run_dir, knobs
        self.db = run_dir / "store.sqlite3"
        self.port = free_port()
        self.token = secrets.token_hex(32)
        self.proc: subprocess.Popen | None = None
        self.url = f"http://127.0.0.1:{self.port}/mcp"
        self.started_at = ""
        self.ready_s: float | None = None
        self.tools: list[dict[str, Any]] = []

    def start(self) -> None:
        assert self.port != 8765
        self.run_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "--reflink=auto", str(EXP / "snap" / f"{self.store}.base.sqlite3"), str(self.db)], check=True)
        env = {
            "HOME": os.environ["HOME"],
            "PATH": os.environ["PATH"],
            "LANG": "C.UTF-8",
            "PYTHONPATH": str(WT / "src"),
            "LM_EXPLICIT_FEEDBACK_PROMPT": self.arm,
            "LM_EXPLICIT_FEEDBACK_POLICY": "audit",
            "LM_DECAY_SWEEP_INTERVAL_SEC": "0",
            "LM_AUTH_TOKEN": self.token,
            **self.knobs,
        }
        cmd = [
            sys.executable, "-m", "living_memory.server", "--transport", "http", "--host", "127.0.0.1",
            "--port", str(self.port), "--db", str(self.db), "--default-scope", "global",
        ]
        self.log = open(self.run_dir / "server.log", "w")
        self.started_at = now()
        self.t0 = time.monotonic()
        self.proc = subprocess.Popen(cmd, env=env, cwd=str(self.run_dir), stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)

    def wait_ready(self) -> bool:
        deadline = self.t0 + READY_TIMEOUT
        while time.monotonic() < deadline:
            if self.proc and self.proc.poll() is not None:
                return False
            try:
                self.tools = asyncio.run(asyncio.wait_for(_list_tools(self.url, self.token), 25))
                self.ready_s = round(time.monotonic() - self.t0, 1)
                return True
            except Exception:
                time.sleep(2)
        return False

    def alive(self) -> bool:
        return bool(self.proc) and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(10)
        self.log.close()
        with _PORT_LOCK:
            _PORTS_USED.discard(self.port)


# ----------------------------------------------------------------------- agents


def agent_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "HOME": os.environ["HOME"],
        "PATH": os.environ["PATH"],
        "USER": os.environ.get("USER", "sfx"),
        "LANG": "C.UTF-8",
    }
    env.update(extra or {})
    return env


def run_claude(sandbox: Sandbox, prompt: str, run_dir: Path, timeout: int = RUN_TIMEOUT) -> dict[str, Any]:
    cwd = run_dir / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    mcp = {"mcpServers": {"living-memory": {"type": "http", "url": sandbox.url, "headers": {"Authorization": f"Bearer {sandbox.token}"}}}}
    (run_dir / "mcp.json").write_text(json.dumps(mcp))
    allowed = ",".join([f"mcp__living-memory__{t}" for t in AGENT_TOOLS] + ["Read", "Grep", "Glob"])
    cmd = [
        "timeout", str(timeout), "claude", "-p", prompt, "--model", CLAUDE_MODEL,
        "--strict-mcp-config", "--mcp-config", str(run_dir / "mcp.json"),
        "--tools", "Read,Grep,Glob", "--allowedTools", allowed,
        "--permission-mode", "dontAsk", "--no-session-persistence",
        "--output-format", "stream-json", "--verbose",
    ]
    env = agent_env({"TERM": "dumb", "ENABLE_CLAUDEAI_MCP_SERVERS": "false"})
    t0 = time.monotonic()
    with open(run_dir / "claude.stream.jsonl", "w") as out, open(run_dir / "claude.stderr", "w") as err:
        code = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=out, stderr=err).returncode
    (run_dir / "mcp.json").unlink()  # holds the sandbox token
    return {"exit": code, "duration_s": round(time.monotonic() - t0, 1), **parse_claude(run_dir / "claude.stream.jsonl", run_dir / "claude.stderr")}


def parse_claude(stream: Path, stderr: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"tool_calls": [], "v2_ok": None, "final_text": "", "usage": {}, "first_input_tokens": None, "is_error": None}
    for line in stream.read_text(errors="replace").splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = msg.get("type")
        if kind == "system" and msg.get("subtype") == "init":
            servers = msg.get("mcp_servers") or []
            tools = msg.get("tools") or []
            info["init_mcp_servers"] = servers
            info["init_tools"] = tools
            info["model"] = msg.get("model")
            info["v2_ok"] = (
                [s.get("name") for s in servers] == ["living-memory"]
                and all(s.get("status") == "connected" for s in servers)
                and all(t in ("Read", "Grep", "Glob") or t.startswith("mcp__living-memory__") for t in tools)
            )
        elif kind == "assistant":
            message = msg.get("message") or {}
            usage = message.get("usage") or {}
            if info["first_input_tokens"] is None and usage:
                info["first_input_tokens"] = sum(int(usage.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
            for block in message.get("content") or []:
                if block.get("type") == "tool_use":
                    info["tool_calls"].append({"name": block.get("name"), "input": block.get("input")})
        elif kind == "result":
            info["final_text"] = msg.get("result") or ""
            info["usage"] = msg.get("usage") or {}
            info["is_error"] = msg.get("is_error")
            info["num_turns"] = msg.get("num_turns")
    usage = info["usage"]
    info["tokens_in"] = sum(int(usage.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    info["tokens_out"] = int(usage.get("output_tokens") or 0)
    err_text = stderr.read_text(errors="replace")[-2000:] if stderr.exists() else ""
    info["provider_error"] = bool(PROVIDER_ERR.search(err_text)) or (bool(info["is_error"]) and bool(PROVIDER_ERR.search(info["final_text"])))
    info["stderr_tail"] = err_text[-400:]
    return info


def run_codex(sandbox: Sandbox, prompt: str, run_dir: Path, timeout: int = RUN_TIMEOUT) -> dict[str, Any]:
    cwd = run_dir / "cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    cmd = [
        "timeout", str(timeout), "codex", "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
        "-s", "read-only", "-C", str(cwd),
        "-c", f'mcp_servers.living-memory.url="http://127.0.0.1:{sandbox.port}/mcp"',
        "-c", 'mcp_servers.living-memory.bearer_token_env_var="LM_SANDBOX_TOKEN"',
        "-c", 'mcp_servers.living-memory.default_tools_approval_mode="approve"',
        "--json", "-o", str(run_dir / "codex.last.txt"), prompt,
    ]
    env = agent_env({"LM_SANDBOX_TOKEN": sandbox.token})
    t0 = time.monotonic()
    with open(run_dir / "codex.events.jsonl", "w") as out, open(run_dir / "codex.stderr", "w") as err:
        code = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL).returncode
    return {"exit": code, "duration_s": round(time.monotonic() - t0, 1), **parse_codex(run_dir, sandbox.port)}


def parse_codex(run_dir: Path, port: int) -> dict[str, Any]:
    info: dict[str, Any] = {"tool_calls": [], "tokens_in": 0, "tokens_out": 0, "first_input_tokens": None, "model": None}
    bad_server = False
    for line in (run_dir / "codex.events.jsonl").read_text(errors="replace").splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("model"):
            info["model"] = msg["model"]
        item = msg.get("item") or {}
        if msg.get("type") == "item.completed" and item.get("type") == "mcp_tool_call":
            server = item.get("server")
            bad_server |= server != "living-memory"
            info["tool_calls"].append({"name": f"mcp__{server}__{item.get('tool')}", "input": item.get("arguments"), "status": item.get("status"), "error": item.get("error")})
        elif msg.get("type") == "item.completed" and item.get("type") in ("command_execution", "web_search"):
            info["tool_calls"].append({"name": item.get("type"), "input": item.get("command") or item.get("query")})
        usage = msg.get("usage")
        if msg.get("type") == "turn.completed" and usage:
            info["tokens_in"] += int(usage.get("input_tokens") or 0)
            info["tokens_out"] += int(usage.get("output_tokens") or 0)
            info["usage"] = usage
    stderr = (run_dir / "codex.stderr").read_text(errors="replace")
    urls = set(re.findall(r"https?://[^\s\"')]+", stderr))
    foreign = sorted(u for u in urls if not u.startswith(f"http://127.0.0.1:{port}") and "openai" not in u and "chatgpt" not in u)
    info["stderr_foreign_urls"] = foreign
    info["v2_ok"] = not bad_server and not foreign
    last = run_dir / "codex.last.txt"
    info["final_text"] = last.read_text(errors="replace") if last.exists() else ""
    info["provider_error"] = bool(PROVIDER_ERR.search(stderr[-3000:])) and not info["tool_calls"]
    info["stderr_tail"] = stderr[-400:]
    return info


# ---------------------------------------------------------------------- export

EXPORT_TABLES = (
    ("recall_events", "created_at"),
    ("recall_feedback_marks", None),
    ("recall_credit_ledger", "credited_at"),
    ("recall_lookup_events", "occurred_at"),
    ("connections", "created_at"),
)


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"pragma table_info({table})")]


def export_run(db: Path, out_dir: Path, since: str) -> dict[str, int]:
    """Rows written after the server started. Nodes: content kept, embeddings dropped."""

    out_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    counts = {}
    try:
        for table, column in (*EXPORT_TABLES, ("nodes", "created_at")):
            cols = _columns(conn, table)
            if not cols:
                counts[table] = -1
                continue
            if column and column not in cols:
                column = next((c for c in ("created_at", "credited_at", "occurred_at") if c in cols), None)
            sql = f"select * from {table}" + (f" where {column} >= ?" if column else "")
            rows = conn.execute(sql, (since,) if column else ()).fetchall()
            with open(out_dir / f"{table}.jsonl", "w") as handle:
                for row in rows:
                    record = {k: row[k] for k in row.keys() if not isinstance(row[k], (bytes, memoryview))}
                    handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            counts[table] = len(rows)
    finally:
        conn.close()
    return counts


# ------------------------------------------------------------------ per-run scoring


def score_run(db: Path, out_dir: Path, since: str, extra_text: str) -> dict[str, Any]:
    """Observations for every sandbox event: G, L, twin G (agreement script) + marks + R_noclose."""

    import explicit_feedback_agreement as ag
    from living_memory.grounding import ground_token_sets, token_set

    conn = ag.open_readonly(db)
    try:
        store = ag.load_store(conn, None)
        since_n = ag.normalize_ts(since)
        store.window_ids = sorted(
            (e.id for e in store.events.values() if e.created_at >= since_n and e.results),
            key=lambda i: store.events[i].created_at,
        )
        observations, stats = ag.observe(conn, store)
        new_nodes = {
            r[0]: (ag.normalize_ts(r[2]), r[1] or "")
            for r in conn.execute("select id, content, created_at from nodes where created_at >= ?", (since,))
        }
        marks = [m for m in (store.marks or [])]
        rows = []
        for o in observations:
            ev = store.events[o.event_id]
            later_nodes = [c for n, (t, c) in new_nodes.items() if t >= ev.created_at and n != o.node_id]
            rows.append({
                "event_id": o.event_id,
                "event_created_at": ev.created_at,
                "query": ev.query,
                "node_id": o.node_id,
                "rank": o.rank,
                "closed": o.closed,
                "closing_trace_id": ev.trace_id,
                "lookup": o.lookup,
                "grounded": o.grounded,
                "containment": o.containment,
                "has_twin": o.has_twin,
                "twin_grounded": o.twin_grounded,
                "cos_trace": o.cos_trace,
                "_later_text": "\n".join(later_nodes),
            })
        # R_noclose: does the node ground against anything the run produced after the event?
        node_tokens: dict[str, frozenset[str]] = {}
        for row in rows:
            nid = row["node_id"]
            if nid not in node_tokens:
                got = conn.execute("select content from nodes where id = ?", (nid,)).fetchone()
                node_tokens[nid] = token_set(got[0] or "") if got else frozenset()
            ref = token_set(row.pop("_later_text") + "\n" + extra_text)
            graded = ground_token_sets(ref, {nid: node_tokens[nid]}, min_containment=MIN_CONTAINMENT) if node_tokens[nid] else {}
            row["grounds_later_work"] = bool(graded and graded[nid].grounded)
        all_events = [
            {"id": e.id, "created_at": e.created_at, "n_results": len(e.results), "trace_id": e.trace_id, "query": e.query}
            for e in store.events.values() if e.created_at >= since_n
        ]
        lookups_after = [(t, n) for items in store.lookups.values() for t, n in items if t >= since_n]
        new_node_times = sorted(t for t, _ in new_nodes.values())
    finally:
        conn.close()
    with open(out_dir / "observations.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "observe_stats": stats,
        "events": all_events,
        "marks": [m.__dict__ for m in marks],
        "lookup_times": sorted(t for t, _ in lookups_after),
        "new_node_times": new_node_times,
    }
    dump_json(out_dir / "scoring.json", summary)
    # The committed agreement script's own report on this sandbox DB (it drops
    # A/B-looking sessions and uses its own n thresholds; decision numbers
    # come from the prereg computation in `analyze`).
    subprocess.run(
        [sys.executable, str(WT / "scripts" / "explicit_feedback_agreement.py"), "--db", str(db), "--host-label",
         out_dir.name, "--since", since, "--json", str(out_dir / "agreement_script.json")],
        capture_output=True, text=True, timeout=900,
    )
    return {"observations": len(rows), "marks": len(marks)}


# ------------------------------------------------------------------------ runs


class Budget:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.tokens = 0
        self.output = 0
        self.fail_streak: Counter[str] = Counter()
        self.stopped_agents: set[str] = set()
        self.abort: str | None = None

    def add(self, info: dict[str, Any]) -> None:
        with self.lock:
            self.tokens += int(info.get("tokens_in") or 0) + int(info.get("tokens_out") or 0)
            self.output += int(info.get("tokens_out") or 0)

    def exceeded(self) -> bool:
        return self.tokens > TOKEN_BUDGET or self.output > OUTPUT_BUDGET


def first_tool_seen(info: dict[str, Any]) -> bool:
    return bool(info.get("tool_calls"))


def one_run(task: dict[str, Any], agent: str, arm: str, knobs: dict[str, str], arm_sha: str, t_exp0: float, attempt: int = 0) -> dict[str, Any]:
    run_id = f"{task['task_id']}__{agent}__{arm}"
    run_dir = EXP / "runs" / (run_id + (f"__retry{attempt}" if attempt else ""))
    if run_dir.exists():
        shutil.rmtree(run_dir)
    sandbox = Sandbox(task["store"], arm, run_dir, knobs)
    record: dict[str, Any] = {"run_id": run_id, "task_id": task["task_id"], "store": task["store"], "agent": agent, "arm": arm, "attempt": attempt, "class": task["class"]}
    sandbox.start()
    try:
        ready = sandbox.wait_ready()
        record.update({"server_started_at": sandbox.started_at, "ready_s": sandbox.ready_s, "port": sandbox.port})
        if not ready:
            record.update({"valid": False, "invalid": "V3 server not ready"})
            return record
        payload = payload_stats(sandbox.tools)
        record["payload_sha256"] = payload["sha256"]
        remaining = int(KILL_AFTER - (time.monotonic() - t_exp0))
        timeout = max(30, min(RUN_TIMEOUT, remaining))
        record["launched_at"] = now()
        info = (run_claude if agent == "claude" else run_codex)(sandbox, task["prompt"], run_dir, timeout)
        record["server_alive_after"] = sandbox.alive()
    finally:
        sandbox.stop()
    record["ended_at"] = now()
    for key in ("exit", "duration_s", "tokens_in", "tokens_out", "first_input_tokens", "v2_ok", "model", "provider_error", "num_turns", "stderr_foreign_urls"):
        record[key] = info.get(key)
    record["tool_calls"] = [c.get("name") for c in info.get("tool_calls", [])]
    record["final_text"] = info.get("final_text", "")[:4000]
    invalid = None
    if info.get("v2_ok") is False:
        invalid = "V2 non-sandbox MCP server or tool in startup/tool calls"
    elif record["payload_sha256"] != arm_sha:
        invalid = "V4 payload sha differs from the arm's"
    elif not record["server_alive_after"]:
        invalid = "V3 server died mid-run"
    elif info.get("exit") == 124 and time.monotonic() - t_exp0 >= KILL_AFTER - 5:
        invalid = "V5 incomplete (S1 kill at 120 min)"
    elif (info.get("exit") not in (0, 124)) or (info.get("exit") == 124 and not first_tool_seen(info)) or info.get("provider_error"):
        invalid = f"V1 exit={info.get('exit')} provider_error={info.get('provider_error')}"
    record["timed_out"] = info.get("exit") == 124
    record["valid"] = invalid is None
    record["invalid"] = invalid
    out_dir = RESULTS / "runs" / (run_id + (f"__retry{attempt}" if attempt else ""))
    record["export_counts"] = export_run(sandbox.db, out_dir, sandbox.started_at)
    extra = record["final_text"] + "\n" + "\n".join(
        json.dumps(c.get("input"), ensure_ascii=False) for c in info.get("tool_calls", []) if "living-memory" not in str(c.get("name"))
    )
    try:
        record["scoring"] = score_run(sandbox.db, out_dir, sandbox.started_at, extra)
    except Exception as exc:  # noqa: BLE001 - recorded, run stays in results
        record["scoring_error"] = repr(exc)[:500]
    # raw agent streams stay in $EFX_EXP_DIR; the repo keeps the parsed record
    for tool_call in info.get("tool_calls", []):
        tool_call["input"] = tool_call.get("input")
    dump_json(out_dir / "tool_calls.json", info.get("tool_calls", []))
    dump_json(out_dir / "run.json", record)
    sandbox.db.unlink(missing_ok=True)
    for suffix in ("-wal", "-shm"):
        Path(str(sandbox.db) + suffix).unlink(missing_ok=True)
    return record


def run_with_retry(task, agent, arm, knobs, arm_sha, t_exp0) -> dict[str, Any]:
    record = one_run(task, agent, arm, knobs, arm_sha, t_exp0)
    if not record["valid"] and str(record.get("invalid", "")).startswith("V1") and time.monotonic() - t_exp0 < NO_LAUNCH_AFTER:
        retry = one_run(task, agent, arm, knobs, arm_sha, t_exp0, attempt=1)
        retry["first_attempt_invalid"] = record["invalid"]
        return retry
    return record


def cmd_run(args: argparse.Namespace) -> int:
    tasks = load_tasks()
    snap = load_json(EXP / "snapshot.json")
    pre = load_json(EXP / "preflight.json")
    shas = {arm: pre["arms"][arm]["sha256"] for arm in ARMS}
    agents_ok = {a for a in ("claude", "codex") if pre["agents_usable"].get(a)}
    stores_ok = {s for s in ("sfx", "alt") if isinstance(snap.get(f"{s}_base"), dict)}
    if args.cells:
        wanted = set(args.cells.split(","))
    else:
        wanted = {f"{s}-{a}" for s, a in CELL_ORDER}
    queue = []
    for position in range(1, 13):
        for store, agent in CELL_ORDER:
            if store not in stores_ok or agent not in agents_ok or f"{store}-{agent}" not in wanted:
                continue
            task = next(t for t in tasks if t["store"] == store and t["launch_position"] == position)
            queue.append((task, agent))
    budget = Budget()
    t_exp0 = time.monotonic()
    state = {"first_launch_at": now(), "queue": [f"{t['task_id']}/{a}" for t, a in queue], "pairs": []}
    lock = threading.Lock()
    head0 = git("rev-parse", "HEAD")

    def pair_job(task: dict[str, Any], agent: str) -> None:
        first = task["first_launched_arm"]
        order = [first, "mandatory" if first == "optional" else "optional"]
        knobs = snap["knobs"][task["store"]]
        with ThreadPoolExecutor(2) as pool:
            fut_a = pool.submit(run_with_retry, task, agent, order[0], knobs, shas[order[0]], t_exp0)
            time.sleep(2)
            fut_b = pool.submit(run_with_retry, task, agent, order[1], knobs, shas[order[1]], t_exp0)
            records = [fut_a.result(), fut_b.result()]
        for record in records:
            budget.add(record)
        provider_fail = all(r.get("provider_error") for r in records)
        with lock:
            budget.fail_streak[agent] = budget.fail_streak[agent] + 1 if provider_fail else 0
            if budget.fail_streak[agent] >= 3:
                budget.stopped_agents.add(agent)
            if any(str(r.get("invalid", "")).startswith("V2") for r in records):
                budget.abort = "S4: V2 showed a non-sandbox LM server"
            state["pairs"].append({"task_id": task["task_id"], "agent": agent, "runs": [r["run_id"] for r in records], "valid": [r["valid"] for r in records], "ended_at": now()})
            dump_json(RESULTS / "run_state.json", {**state, "tokens": budget.tokens, "output_tokens": budget.output})
        print(f"[{now()}] pair {task['task_id']}/{agent}: " + ", ".join(f"{r['arm']}={'ok' if r['valid'] else r['invalid']}" for r in records), flush=True)

    stop_reason = None
    with ThreadPoolExecutor(MAX_PAIRS) as pool:
        futures = []
        for task, agent in queue:
            while True:
                running = sum(1 for f in futures if not f.done())
                if running < MAX_PAIRS:
                    break
                time.sleep(3)
            elapsed = time.monotonic() - t_exp0
            if elapsed > NO_LAUNCH_AFTER:
                stop_reason = "S1: no launch after 105 min"
                break
            if budget.exceeded():
                stop_reason = "S2: token budget"
                break
            if budget.abort:
                stop_reason = budget.abort
                break
            if agent in budget.stopped_agents:
                continue
            if git("rev-parse", "HEAD") != head0:
                stop_reason = "X4: HEAD moved"
                break
            futures.append(pool.submit(pair_job, task, agent))
            time.sleep(1)
        for future in futures:
            future.result()
    state.update({"finished_at": now(), "stop_reason": stop_reason, "tokens": budget.tokens, "output_tokens": budget.output, "stopped_agents": sorted(budget.stopped_agents), "abort": budget.abort, "head": head0})
    dump_json(RESULTS / "run_state.json", state)
    print(json.dumps({k: state[k] for k in ("finished_at", "stop_reason", "tokens", "output_tokens", "stopped_agents", "abort")}, indent=2))
    return 0


# ------------------------------------------------------------------- preflight


def cmd_preflight(args: argparse.Namespace) -> int:
    snap = load_json(EXP / "snapshot.json")
    out: dict[str, Any] = {"arms": {}, "smoke": {}, "agents_usable": {}, "at": now()}
    for arm in ARMS:
        run_dir = EXP / "preflight" / arm
        if run_dir.exists():
            shutil.rmtree(run_dir)
        sandbox = Sandbox("sfx", arm, run_dir, snap["knobs"]["sfx"])
        sandbox.start()
        try:
            ready = sandbox.wait_ready()
            stats = payload_stats(sandbox.tools) if ready else {}
            stats["ready_s"] = sandbox.ready_s
            dump_json(RESULTS / "preflight" / f"tools_list_{arm}.json", sandbox.tools)
            out["arms"][arm] = stats
            if ready and not args.no_smoke:
                for agent in ("claude", "codex"):
                    agent_dir = run_dir / agent
                    agent_dir.mkdir(parents=True, exist_ok=True)
                    info = (run_claude if agent == "claude" else run_codex)(sandbox, SMOKE_PROMPT, agent_dir, 180)
                    recall_ok = any(
                        str(c.get("name", "")).endswith("memory_recall") and c.get("error") in (None, "") and c.get("status", "completed") in ("completed", None)
                        for c in info.get("tool_calls", [])
                    )
                    smoke = {
                        "exit": info.get("exit"),
                        "duration_s": info.get("duration_s"),
                        "v2_ok": info.get("v2_ok"),
                        "recall_call_ok": recall_ok,
                        "tool_calls": [c.get("name") for c in info.get("tool_calls", [])],
                        "init_mcp_servers": info.get("init_mcp_servers"),
                        "init_tools": info.get("init_tools"),
                        "model": info.get("model"),
                        "stderr_foreign_urls": info.get("stderr_foreign_urls"),
                        "final_text": (info.get("final_text") or "")[:300],
                        "stderr_tail": info.get("stderr_tail"),
                    }
                    out["smoke"][f"{arm}/{agent}"] = smoke
        finally:
            sandbox.stop()
    out["payloads_differ"] = out["arms"]["optional"].get("sha256") != out["arms"]["mandatory"].get("sha256")
    for agent in ("claude", "codex"):
        smokes = [out["smoke"].get(f"{arm}/{agent}", {}) for arm in ARMS]
        out["agents_usable"][agent] = all(
            s.get("exit") == 0 and s.get("recall_call_ok") and s.get("v2_ok") and (s.get("duration_s") or 999) <= (120 if agent == "codex" else 420)
            for s in smokes
        ) if smokes and all(smokes) else False
    dump_json(EXP / "preflight.json", out)
    dump_json(RESULTS / "preflight" / "preflight.json", out)
    print(json.dumps(out, indent=2)[:6000])
    return 0


# ------------------------------------------------------------------- leakcheck


def cmd_leakcheck(_: argparse.Namespace) -> int:
    state = load_json(RESULTS / "run_state.json")
    start, end = state["first_launch_at"], state.get("finished_at") or now()
    queries = set()
    run_sessions = []
    for run_json in (RESULTS / "runs").glob("*/run.json"):
        for line in (run_json.parent / "recall_events.jsonl").read_text().splitlines():
            row = json.loads(line)
            queries.add(row["query"])
            run_sessions.append(row.get("transport_session_id"))
    task_queries = {t["query"] for t in load_tasks()}
    out: dict[str, Any] = {"window": [start, end], "sandbox_queries": len(queries), "checked_at": now()}
    code = (
        "import sqlite3,json,sys;q=json.load(sys.stdin);c=sqlite3.connect('file:{db}?mode=ro',uri=True);"
        "rows=c.execute('select id,query,agent,transport_session_id,created_at from recall_events where created_at>=? and created_at<=?',(q['s'],q['e'])).fetchall();"
        "qs=set(q['q']);print(json.dumps({{'window_events':len(rows),'matches':[r for r in rows if r[1] in qs]}}))"
    )
    payload = json.dumps({"s": start, "e": end, "q": sorted(queries)})
    for store in ("sfx", "alt"):
        py = code.format(db=LIVE[store])
        if store == "sfx":
            res = subprocess.run([sys.executable, "-c", py], input=payload, capture_output=True, text=True, timeout=300)
        else:
            res = subprocess.run(["timeout", "300", "ssh", "alt", f"{ALT_PYTHON} -c \"{py}\""], input=payload, capture_output=True, text=True)
        try:
            found = json.loads(res.stdout)
        except json.JSONDecodeError:
            out[store] = {"error": res.stderr[-300:]}
            continue
        matches = found["matches"]
        unexplained = [m for m in matches if m[1] not in task_queries or m[3] not in run_sessions]
        out[store] = {
            "window_events": found["window_events"],
            "exact_query_matches": len(matches),
            "matches": [{"id": m[0], "agent": m[2], "created_at": m[4], "is_task_query": m[1] in task_queries} for m in matches],
            "x1_live_touch": bool(matches),
        }
        out[store]["recall_events_after"] = live_recall_count(store)
    dump_json(RESULTS / "leakcheck.json", out)
    print(json.dumps(out, indent=2))
    return 0


# --------------------------------------------------------------------- analyze


def _perm_test(events: list[dict[str, Any]], stat_fn, rng: np.random.Generator) -> tuple[float, float]:
    observed = stat_fn(events, None)
    ge = 0
    for _ in range(PERM_ROUNDS):
        perms = [rng.permutation(len(e["s"])) for e in events]
        if stat_fn(events, perms) >= observed - 1e-12:
            ge += 1
    return observed, (1 + ge) / (PERM_ROUNDS + 1)


def d_stat(label: str, sign: int):
    """D = sign * (mean s(labelled) - mean s(unlabelled)), pairs pooled over events."""

    def stat(events, perms):
        on, off = [], []
        for i, e in enumerate(events):
            labels = e["labels"] if perms is None else [e["labels"][j] for j in perms[i]]
            for s, lab in zip(e["s"], labels):
                (on if lab == label else off).append(s)
        if not on or not off:
            return float("nan")
        return sign * (float(np.mean(on)) - float(np.mean(off)))

    return stat


def agreement(events: list[dict[str, Any]], label: str, sign: int, min_marks: int = 20, min_events: int = 8) -> dict[str, Any]:
    events = [e for e in events if label in e["labels"] and len(e["s"]) >= 2]
    n_marks = sum(e["labels"].count(label) for e in events)
    out: dict[str, Any] = {"n_marks": n_marks, "n_events": len(events)}
    if not events:
        out.update({"D": None, "p": None, "ci95": [None, None], "better_than_random": False, "status": "no marks"})
        return out
    stat = d_stat(label, sign)
    rng = np.random.default_rng(SEED)
    observed, p = _perm_test(events, stat, rng)
    boot_rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(PERM_ROUNDS):
        sample = [events[i] for i in boot_rng.integers(len(events), size=len(events))]
        value = stat(sample, None)
        if not np.isnan(value):
            boots.append(value)
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else [None, None]
    enough = n_marks >= min_marks and len(events) >= min_events
    btr = bool(enough and observed > 0 and p < 0.05)
    out.update({
        "D": None if np.isnan(observed) else round(observed, 4),
        "p": round(p, 4),
        "ci95": [None if v is None else round(v, 4) for v in ci],
        "better_than_random": btr,
        "status": "decided" if enough else f"inconclusive (<{min_marks} marks or <{min_events} events)",
    })
    return out


def cell_metrics(runs: list[dict[str, Any]], tasks_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    events_all = 0
    marked = 0
    markable = 0
    marked_markable = 0
    used_only = irr_only = both = 0
    marks_per_marked: list[int] = []
    accepted_used = rejected = accepted_irr = total_marks = 0
    eval_events: list[dict[str, Any]] = []
    grounded_events: list[dict[str, Any]] = []  # descriptive: closed events, s = G - G(twin), no lookup
    rit = Counter()
    used_marks_all = 0
    noclose = 0
    gold = {"events_with_gold_delivered": 0, "used_on_gold": 0, "used_marks_in_those": 0}
    cls_used = {"answer_in_memory": [0, 0], "answer_not_in_memory": [0, 0]}
    known_irr = Counter()
    recalls_per_run = []
    zero_recall_runs = 0
    dropped_no_twin = 0
    for run in runs:
        rdir = RESULTS / "runs" / run["dir"]
        scoring = load_json(rdir / "scoring.json", {})
        obs = [json.loads(l) for l in (rdir / "observations.jsonl").read_text().splitlines()] if (rdir / "observations.jsonl").exists() else []
        task = tasks_by_id[run["task_id"]]
        events = sorted(scoring.get("events", []), key=lambda e: e["created_at"])
        recalls_per_run.append(len(events))
        zero_recall_runs += not events
        marks = scoring.get("marks", [])
        total_marks += len(marks)
        rejected += sum(1 for m in marks if not m["accepted"])
        acc = [m for m in marks if m["accepted"]]
        by_event: dict[str, dict[str, set]] = defaultdict(lambda: {"used": set(), "irrelevant": set()})
        for m in acc:
            by_event[m["event_id"]][m["mark"]].add(m["node_id"])
        accepted_used += sum(1 for m in acc if m["mark"] == "used")
        accepted_irr += sum(1 for m in acc if m["mark"] == "irrelevant")
        later_calls = sorted([e["created_at"] for e in events] + scoring.get("lookup_times", []) + scoring.get("new_node_times", []))
        obs_by_event: dict[str, list[dict]] = defaultdict(list)
        for o in obs:
            obs_by_event[o["event_id"]].append(o)
        run_used_events = []
        for e in events:
            events_all += 1
            m = by_event.get(e["id"])
            has = bool(m and (m["used"] or m["irrelevant"]))
            marked += has
            is_markable = any(t > e["created_at"] for t in later_calls)
            markable += is_markable
            marked_markable += has and is_markable
            if has:
                marks_per_marked.append(len(m["used"]) + len(m["irrelevant"]))
                if m["used"] and m["irrelevant"]:
                    both += 1
                elif m["used"]:
                    used_only += 1
                else:
                    irr_only += 1
            rows = sorted(obs_by_event.get(e["id"], []), key=lambda o: o["rank"])
            used = (m or {}).get("used", set())
            irr = (m or {}).get("irrelevant", set()) - used
            delivered = [o["node_id"] for o in rows]
            if used:
                run_used_events.append((e["id"], {o["rank"] for o in rows if o["node_id"] in used}))
                used_marks_all += len(used)
                cls_used[task["class"]][0] += len(used)
                if len(delivered) >= 3 and set(delivered) == used:
                    rit["R_all"] += len(used)
                    for o in rows:
                        o["_rall"] = True
                noclose += sum(1 for o in rows if o["node_id"] in used and not o.get("grounds_later_work"))
            cls_used[task["class"]][1] += len(delivered)
            gold_ids = set(task["gold_node_ids"])
            if gold_ids & set(delivered):
                gold["events_with_gold_delivered"] += 1
                gold["used_on_gold"] += len(used & gold_ids)
                gold["used_marks_in_those"] += len(used)
            for nid in set(task["known_irrelevant_node_ids"]) & set(delivered):
                known_irr["delivered"] += 1
                known_irr["marked_used"] += nid in used
                known_irr["marked_irrelevant"] += nid in irr
            evaluable = bool(e.get("trace_id")) or any(o["lookup"] for o in rows)
            if not evaluable or not rows:
                continue
            s_vals, labels, g_vals = [], [], []
            for o in rows:
                if o["closed"] and not o["has_twin"]:
                    dropped_no_twin += 1
                    continue
                g = 1.0 if (o["grounded"] or o["lookup"]) else 0.0
                twin = 1.0 if (o["closed"] and o["has_twin"] and o["twin_grounded"]) else 0.0
                s_vals.append(g - twin)
                g_vals.append((1.0 if o["grounded"] else 0.0) - twin)
                labels.append("used" if o["node_id"] in used else "irrelevant" if o["node_id"] in irr else "none")
            if s_vals:
                eval_events.append({"id": e["id"], "s": s_vals, "labels": labels})
                if e.get("trace_id"):
                    grounded_events.append({"id": e["id"], "s": g_vals, "labels": labels})
        # R_rank1 per run
        if len(run_used_events) >= 2 and all(ranks == {0} for _, ranks in run_used_events):
            rit["R_rank1"] += len(run_used_events)  # one used mark per event, all rank 0
            rit["R_rank1_runs"] += 1
        # union R_all ∪ R_rank1: they are disjoint when every event has ≥3 results and only rank0 is used
    ritual_union = rit["R_all"] + rit["R_rank1"]
    d_used = agreement(eval_events, "used", +1)
    d_irr = agreement(eval_events, "irrelevant", -1)

    def share(a, b):
        return None if not b else round(a / b, 4)

    return {
        "runs": len(runs),
        "recall_events": events_all,
        "C": share(marked, events_all),
        "C_markable": share(marked_markable, markable),
        "markable_events": markable,
        "marked_events": marked,
        "used_only_share": share(used_only, events_all),
        "irrelevant_only_share": share(irr_only, events_all),
        "both_share": share(both, events_all),
        "marks_per_marked_event": round(float(np.mean(marks_per_marked)), 2) if marks_per_marked else None,
        "marks_total_rows": total_marks,
        "accepted_used": accepted_used,
        "accepted_irrelevant": accepted_irr,
        "reject_rate": share(rejected, total_marks),
        "evaluable_events": len(eval_events),
        "pairs_dropped_no_twin": dropped_no_twin,
        "D_used": d_used,
        "D_irr": d_irr,
        "D_used_grounded_only_descriptive": agreement(grounded_events, "used", +1),
        "ritual": {
            "used_marks": used_marks_all,
            "R_all": rit["R_all"],
            "R_rank1": rit["R_rank1"],
            "R_rank1_runs": rit["R_rank1_runs"],
            "RS": share(ritual_union, used_marks_all),
            "R_noclose": noclose,
            "R_noclose_share": share(noclose, used_marks_all),
        },
        "gold": {**gold, "gold_hit_rate": share(gold["used_on_gold"], gold["used_marks_in_those"])},
        "used_rate_by_class": {k: share(v[0], v[1]) for k, v in cls_used.items()},
        "known_irrelevant": dict(known_irr),
        "recalls_per_run_mean": round(float(np.mean(recalls_per_run)), 2) if recalls_per_run else None,
        "zero_recall_runs": zero_recall_runs,
    }


def weight(d: dict[str, Any]) -> float:
    lo = d["ci95"][0]
    if lo is None:
        return 0.25
    return 1.0 if lo >= 0.10 else 0.5 if lo > 0 else 0.25


def decide(m_o: dict[str, Any], m_m: dict[str, Any], per_agent: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ag_o = m_o["D_used"]["better_than_random"]
    ag_m = m_m["D_used"]["better_than_random"]
    ag_m_inconclusive = not m_m["D_used"]["status"].startswith("decided")
    used_m = m_m["ritual"]["used_marks"]
    rs_m = m_m["ritual"]["RS"]
    rit_m_inconclusive = used_m < 20
    rit_m = (not rit_m_inconclusive) and rs_m is not None and rs_m > 0.5
    poor_o = m_o["C"] is None or m_o["C"] < 0.10
    c_o, c_m = m_o["C"] or 0.0, m_m["C"] or 0.0
    flags = {"AG_o": ag_o, "AG_m": ag_m, "AG_m_inconclusive": ag_m_inconclusive, "RIT_m": rit_m, "RIT_m_inconclusive": rit_m_inconclusive, "POOR_o": poor_o, "C_o": c_o, "C_m": c_m}
    falsifiers = {
        "F1_no_agreement_either_arm": (not ag_o) and (not ag_m),
        "F2_mandatory_rejected_ritual": rit_m,
        "F3_optional_channel_poor": poor_o,
    }

    def w(arm):
        metrics = m_o if arm == "optional" else m_m
        value = weight(metrics["D_used"])
        halved = [a for a, cells in per_agent.items() if (cells[arm]["D_used"]["n_marks"] >= 10 and (cells[arm]["D_used"]["D"] or 0) <= 0)]
        return (value / 2 if halved else value), halved

    rows = [
        ("F1", (not ag_o) and (not ag_m), ("optional", "audit", None)),
        ("M0", (rit_m_inconclusive and ag_m_inconclusive) and (poor_o or not ag_o), ("optional", "audit", None)),
        ("R1", rit_m and (poor_o or not ag_o), ("optional", "audit", None)),
        ("R2", rit_m and ag_o and not poor_o, ("optional", "credit", "optional")),
        ("P1", poor_o and ag_m and not rit_m, ("mandatory", "credit", "mandatory")),
        ("P2", poor_o and ((not ag_m) or rit_m), ("optional", "audit", None)),
        ("O1", ag_o and not poor_o and ag_m and not rit_m and c_m >= 2 * c_o and c_m - c_o >= 0.15, ("mandatory", "credit", "mandatory")),
        ("O2", ag_o and not poor_o, ("optional", "credit", "optional")),
        ("O3", (not ag_o) and ag_m and not rit_m, ("mandatory", "credit", "mandatory")),
    ]
    for name, cond, (prompt, policy, warm) in rows:
        if cond:
            if warm:
                value, halved = w(warm)
            else:
                value, halved = 0.0, []
            return {"flags": flags, "falsifiers": falsifiers, "row": name, "LM_EXPLICIT_FEEDBACK_PROMPT": prompt, "LM_EXPLICIT_FEEDBACK_POLICY": policy, "LM_EXPLICIT_CREDIT_WEIGHT": value, "weight_halved_for_agents": halved}
    return {"flags": flags, "falsifiers": falsifiers, "row": None}


def cmd_analyze(_: argparse.Namespace) -> int:
    tasks_by_id = {t["task_id"]: t for t in load_tasks()}
    records = []
    for run_json in sorted((RESULTS / "runs").glob("*/run.json")):
        record = load_json(run_json)
        record["dir"] = run_json.parent.name
        records.append(record)
    # final attempt per (task, agent, arm)
    final: dict[tuple, dict] = {}
    for r in sorted(records, key=lambda r: r["attempt"]):
        final[(r["task_id"], r["agent"], r["arm"])] = r
    # pair validity: both arms valid
    valid_pairs = set()
    for (task_id, agent, arm), r in final.items():
        other = final.get((task_id, agent, "mandatory" if arm == "optional" else "optional"))
        if r["valid"] and other and other["valid"]:
            valid_pairs.add((task_id, agent))
    summary: dict[str, Any] = {"generated_at": now(), "cells": {}, "stores": {}, "validity": {}}
    pre = load_json(EXP / "preflight.json", {})
    for store in ("sfx", "alt"):
        store_out: dict[str, Any] = {}
        per_agent: dict[str, dict[str, Any]] = {}
        for agent in ("claude", "codex"):
            cell = [r for (t, a, _), r in final.items() if a == agent and tasks_by_id[t]["store"] == store]
            if not cell:
                continue
            n_invalid = sum(1 for r in cell if not r["valid"])
            pairs = {(t, agent) for (t, a) in valid_pairs if a == agent and tasks_by_id[t]["store"] == store}
            summary["validity"][f"{store}-{agent}"] = {
                "runs": len(cell),
                "invalid_runs": n_invalid,
                "invalid_reasons": Counter(r["invalid"] for r in cell if not r["valid"]),
                "valid_pairs": len(pairs),
                "X3_cell_invalid": n_invalid > 0.25 * len(cell),
                "retried_runs": sum(1 for r in cell if r["attempt"]),
            }
            per_agent[agent] = {}
            for arm in ARMS:
                runs = [r for r in cell if r["arm"] == arm and (r["task_id"], agent) in pairs]
                per_agent[agent][arm] = cell_metrics(runs, tasks_by_id)
            # first-request token delta (claude)
            deltas = []
            for t, _ in pairs:
                a = final[(t, agent, "optional")].get("first_input_tokens")
                b = final[(t, agent, "mandatory")].get("first_input_tokens")
                if a and b:
                    deltas.append(b - a)
            per_agent[agent]["first_request_input_delta_median"] = float(np.median(deltas)) if deltas else None
            per_agent[agent]["tokens_per_run"] = {
                arm: round(float(np.mean([(r.get("tokens_in") or 0) + (r.get("tokens_out") or 0) for r in cell if r["arm"] == arm and (r["task_id"], agent) in pairs] or [0]))) for arm in ARMS
            }
            per_agent[agent]["duration_per_run_s"] = {
                arm: round(float(np.mean([r.get("duration_s") or 0 for r in cell if r["arm"] == arm and (r["task_id"], agent) in pairs] or [0])), 1) for arm in ARMS
            }
            cells_valid = {
                arm: [r for r in cell if r["arm"] == arm and (r["task_id"], agent) in pairs] for arm in ARMS
            }
            per_agent[agent]["answer_outcome"] = {arm: answer_outcome(cells_valid[arm], tasks_by_id) for arm in ARMS}
        if not per_agent:
            summary["stores"][store] = {"status": "not run"}
            continue
        pooled = {}
        for arm in ARMS:
            runs = [
                r for (t, a, ar), r in final.items()
                if ar == arm and tasks_by_id[t]["store"] == store and (t, a) in valid_pairs
                and not summary["validity"][f"{store}-{a}"]["X3_cell_invalid"]
            ]
            pooled[arm] = cell_metrics(runs, tasks_by_id)
        claude_valid = summary["validity"].get(f"{store}-claude", {})
        invalid_reason = None
        if claude_valid.get("X3_cell_invalid"):
            invalid_reason = "X3: Claude cell >25% invalid runs"
        elif claude_valid.get("valid_pairs", 0) < 6:
            invalid_reason = f"X5: {claude_valid.get('valid_pairs', 0)} valid Claude pairs (<6)"
        store_out["pooled"] = pooled
        store_out["by_agent"] = per_agent
        store_out["invalid"] = invalid_reason
        store_out["decision"] = None if invalid_reason else decide(pooled["optional"], pooled["mandatory"], per_agent)
        summary["stores"][store] = store_out
    summary["payload"] = {arm: pre.get("arms", {}).get(arm) for arm in ARMS}
    dump_json(RESULTS / "summary.json", summary)
    print(json.dumps({s: (v.get("decision") if isinstance(v, dict) else v) for s, v in summary["stores"].items()}, indent=2, default=str))
    return 0


def answer_outcome(runs: list[dict[str, Any]], tasks_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    a_hits = a_n = b_nf = b_n = 0
    for r in runs:
        task = tasks_by_id[r["task_id"]]
        text = r.get("final_text") or ""
        if task["class"] == "answer_in_memory":
            a_n += 1
            a_hits += any(g in text for g in task["gold_node_ids"])
        else:
            b_n += 1
            b_nf += bool(re.search(r"not found|could not find|couldn't find|no (relevant )?(record|memory|information)", text, re.I))
    return {"A_names_gold_id": f"{a_hits}/{a_n}", "B_says_not_found": f"{b_nf}/{b_n}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify-base").set_defaults(fn=cmd_verify_base)
    sub.add_parser("snapshot").set_defaults(fn=cmd_snapshot)
    sub.add_parser("prepare").set_defaults(fn=cmd_prepare)
    p = sub.add_parser("preflight")
    p.add_argument("--no-smoke", action="store_true")
    p.set_defaults(fn=cmd_preflight)
    p = sub.add_parser("run")
    p.add_argument("--cells", default=None, help="comma list like sfx-claude,alt-claude (default: all usable)")
    p.set_defaults(fn=cmd_run)
    sub.add_parser("leakcheck").set_defaults(fn=cmd_leakcheck)
    sub.add_parser("analyze").set_defaults(fn=cmd_analyze)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
