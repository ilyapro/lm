#!/usr/bin/env python3
"""attest_eval.py — does grounded-usage attestation discriminate on real sessions?

``memory_attest`` (see :mod:`living_memory.attestation`) is supposed to credit a
finished session's recall events *only* when the session's own artifacts contain
what those events delivered. That claim is falsifiable, and this script falsifies
it or does not, on held-out real transcripts rather than on fixtures:

TRUE pairing
    every eval-split session's evidence graded against **its own** recall events.

SHUFFLED pairing
    the same events graded against **another session's** evidence, under a fixed
    derangement of the session list. Same events, same evidence documents, same
    grader — only the pairing is wrong. Whatever rate survives here is the rate
    at which the mechanism credits noise.

A mechanism that measures use has a high true rate and a near-zero shuffled one.
A mechanism that merely rewards long documents scores the same in both arms, and
the shuffled arm is the only thing that can tell those apart, which is why it
exists.

Everything the run touches is disposable
----------------------------------------
* the database is a **scratch** SQLite file in a temp dir, built here from the
  transcripts. ``MemoryStore.__init__`` migrates and writes whatever file it
  opens, so no offline process may open the live database
  (``~/.local/share/living-memory/global.sqlite3``);
* the server is a **throwaway** ``python -m living_memory.server`` started from
  this worktree on a free port with a freshly generated ``LM_AUTH_TOKEN``, never
  the live unit on 127.0.0.1:8765;
* every attestation goes over MCP into that server, so what is measured is the
  deployed tool contract and not a direct call into the module.

Reconstruction, and why it is conservative
------------------------------------------
The scratch database is rebuilt from what the transcripts recorded: one node per
``DeliveredNode`` (its ``content``), one ``recall_events`` row per recall, with
the same result payloads (rank order and per-channel scores).

Only deliveries with ``content_truncated == False`` become nodes. A snippet has
a **smaller token set** than the node it stands for, and containment is
``|node ∩ evidence| / |node|`` — so grading against a snippet makes the gate
*easier* to clear and would inflate the result in the flattering direction.
Truncated deliveries still occupy their rank slot in the replayed results, so
the rank decay of the surviving ones is unchanged; the attestation path already
skips a result whose node it cannot resolve, exactly as it would for a node
decayed between recall and attestation. The count is reported as
``truncated_deliveries_excluded``.

Split discipline
----------------
The tracked manifest ``artifacts/post-session/corpus.json`` seals each split as
``sha256`` over its sorted session keys; the index that lists the keys is
gitignored and rebuilt from disk, so it drifts as new sessions accumulate. This
script does not trust the ``split`` label on a rebuilt row. It **recovers the
sealed membership**: for each split it searches the rows created after the seal
for the subset whose removal reproduces the sealed digest byte-for-byte. When
that succeeds — it does when the drift is purely additive — the sealed key sets
are known exactly, and "this session was not in the sealed holdout" is a fact
about the seal rather than about a label. When it fails the run aborts, because
a corpus whose splits moved cannot support a held-out claim at all.

``holdout_sessions_read`` is then real accounting, not an assertion: every
transcript this script opens goes through :class:`TranscriptReader`, which looks
the path's session key up in the recovered sealed membership, counts the open
under that split, and raises before opening anything that is not in the split
under measurement.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import itertools
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for _candidate in (REPO_ROOT, SRC_DIR):
    _value = str(_candidate)
    if _value not in sys.path:
        sys.path.insert(0, _value)

from living_memory.attestation import (  # noqa: E402
    ATTESTATION_MIN_CONTAINMENT,
    EVIDENCE_MAX_ITEM_CHARS,
    EVIDENCE_MAX_ITEMS,
    EVIDENCE_MAX_TOTAL_CHARS,
)
from living_memory.postsession.corpus import (  # noqa: E402
    DEFAULT_INDEX,
    DEFAULT_MANIFEST,
    SPLITS,
    SessionEntry,
    read_index,
    read_manifest,
    ref_for,
    split_digest,
)
from living_memory.postsession.evidence import EvidenceBundle, assemble_evidence  # noqa: E402
from living_memory.postsession.session import SessionRecord  # noqa: E402
from living_memory.postsession.transcripts import load_transcript  # noqa: E402
from living_memory.storage import MemoryStore  # noqa: E402

DEFAULT_OUT = Path("artifacts/post-session/attestation-eval.json")

#: Where this script expects the rebuilt session index, and *not*
#: ``postsession.corpus.DEFAULT_INDEX``.
#:
#: ``artifacts/post-session/corpus-index.jsonl`` is the slot
#: ``scripts/postsession_corpus.py --build`` re-seals, and
#: ``tests/test_postsession_corpus.py`` asserts that an index sitting there
#: hashes to the tracked manifest's ``index_sha256``. That assertion is right,
#: and it is exactly what a rebuild cannot satisfy on a machine that has
#: accumulated sessions since the seal — which is every machine, a day later.
#: A read-only consumer must not park a drifted index in the seal's slot to
#: make its own life easier, so it keeps its own copy under the ignored
#: ``.cache/`` tree and proves the relationship to the seal by reconciliation
#: instead (see :func:`recover_sealed_corpus`).
DEFAULT_WORK_INDEX = Path(".cache/post-session/corpus-index.jsonl")

#: Node levels the store accepts; anything else a transcript recorded falls back
#: to ``trace``.
NODE_LEVELS = frozenset({"trace", "concept", "schema"})

#: Containment histogram edges. The gate sits at 0.25, so the buckets around it
#: are the ones that decide, and they are narrow there on purpose.
HISTOGRAM_EDGES: tuple[float, ...] = (
    0.0,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.35,
    0.50,
    0.75,
    1.0,
)

#: Longest the run waits for the throwaway server to answer on its port.
SERVER_BOOT_TIMEOUT = 90.0

#: The live unit. Naming it here is not a default — it is the thing this script
#: must never talk to, and the guard below fails loudly if a port collides.
LIVE_PORT = 8765
LIVE_DB = Path.home() / ".local" / "share" / "living-memory" / "global.sqlite3"


class EvalError(RuntimeError):
    """The measurement cannot be performed as asked."""


# ---------------------------------------------------------------------------
# corpus: recover the sealed split membership
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SealedCorpus:
    """The sealed split membership, recovered from the manifest plus the index."""

    manifest: dict[str, Any]
    manifest_sha256: str
    index_path: str
    index_sha256: str
    rows: dict[str, SessionEntry]
    membership: dict[str, frozenset[str]]
    #: Rows present now that the seal never covered, per split.
    post_seal: dict[str, tuple[str, ...]]

    def split_of(self, session_key: str) -> str | None:
        for name, keys in self.membership.items():
            if session_key in keys:
                return name
        return None

    def report(self) -> dict[str, Any]:
        return {
            "manifest": str(DEFAULT_MANIFEST),
            "manifest_sha256": self.manifest_sha256,
            "working_index": self.index_path,
            "working_index_sha256": self.index_sha256,
            "working_index_note": (
                "rebuilt from disk; deliberately not written to "
                f"{DEFAULT_INDEX}, which is the seal's own slot"
            ),
            "manifest_generated_at": self.manifest.get("generated_at"),
            "manifest_index_sha256": self.manifest.get("index_sha256"),
            "manifest_holdout_sha256": self.manifest.get("holdout_sha256"),
            "sealed_split_sha256": dict(self.manifest.get("split_sha256") or {}),
            "sealed_sizes": {name: len(keys) for name, keys in self.membership.items()},
            "sealed_membership_recovered": True,
            "post_seal_rows_excluded": {
                name: len(keys) for name, keys in self.post_seal.items()
            },
        }


def _post_seal_candidates(rows: Sequence[SessionEntry], seal_at: float) -> list[str]:
    """Keys whose transcript file was created/last written after the seal.

    Coarse on purpose: it only has to be a *superset* of the rows the seal never
    saw, because the exact subset is then found by digest.
    """

    fresh: list[str] = []
    for row in rows:
        try:
            mtime = os.stat(row.path).st_mtime
        except OSError:
            continue
        if mtime > seal_at:
            fresh.append(row.session_key)
    return sorted(fresh)


def recover_sealed_corpus(manifest_path: Path, index_path: Path) -> SealedCorpus:
    """Sealed per-split key sets, proven against the tracked digests."""

    if not manifest_path.is_file():
        raise EvalError(f"manifest not found: {manifest_path}")
    if not index_path.is_file():
        raise EvalError(
            f"index not found: {index_path} — rebuild it with\n"
            "  python3 scripts/postsession_corpus.py --build \\\n"
            "      --manifest /tmp/corpus-rebuild.json \\\n"
            f"      --index {index_path}\n"
            "(both outputs are throwaway: the tracked manifest is the seal and "
            "must not be rewritten by a read-only consumer)"
        )
    manifest = read_manifest(manifest_path)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    index_sha256 = hashlib.sha256(index_path.read_bytes()).hexdigest()
    sealed_digests = manifest.get("split_sha256") or {}
    sealed_counts = (manifest.get("counts") or {}).get("by_split") or {}
    generated_at = str(manifest.get("generated_at") or "")
    try:
        seal_at = datetime.fromisoformat(generated_at.replace("Z", "+00:00")).timestamp()
    except ValueError as error:
        raise EvalError(f"manifest generated_at is unusable: {generated_at!r}") from error

    rows = {row.session_key: row for row in read_index(index_path)}
    membership: dict[str, frozenset[str]] = {}
    post_seal: dict[str, tuple[str, ...]] = {}
    for name in SPLITS:
        current = sorted(
            (row for row in rows.values() if row.split == name),
            key=lambda row: row.session_key,
        )
        keys = [row.session_key for row in current]
        target = sealed_digests.get(name)
        if not target:
            raise EvalError(f"manifest carries no sealed digest for split {name!r}")
        if split_digest(keys) == target:
            membership[name] = frozenset(keys)
            post_seal[name] = ()
            continue
        surplus = len(keys) - int(sealed_counts.get(name, 0))
        candidates = _post_seal_candidates(current, seal_at)
        recovered = _reconcile(keys, candidates, surplus, target)
        if recovered is None:
            raise EvalError(
                f"split {name!r} no longer matches its seal and the difference is not "
                f"purely additive: {len(keys)} rows now vs {sealed_counts.get(name)} sealed, "
                f"{len(candidates)} written after the seal. A corpus whose splits moved "
                "cannot support a held-out claim; re-seal it deliberately instead."
            )
        membership[name] = frozenset(recovered[0])
        post_seal[name] = recovered[1]
    return SealedCorpus(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        index_path=str(index_path),
        index_sha256=index_sha256,
        rows=rows,
        membership=membership,
        post_seal=post_seal,
    )


def _reconcile(
    keys: Sequence[str], candidates: Sequence[str], surplus: int, target: str
) -> tuple[list[str], tuple[str, ...]] | None:
    """Find the post-seal subset whose removal reproduces ``target``.

    ``surplus`` is how many rows the split gained, so only subsets of that size
    can be the answer; ``candidates`` is the (small) superset of rows written
    after the seal. Returns the sealed key list and the removed keys.
    """

    if surplus < 0 or surplus > len(candidates):
        return None
    present = set(keys)
    for combo in itertools.combinations(candidates, surplus):
        dropped = set(combo)
        rest = [key for key in keys if key not in dropped]
        if split_digest(rest) == target:
            return rest, tuple(sorted(dropped & present))
    return None


# ---------------------------------------------------------------------------
# reading transcripts, under accounting
# ---------------------------------------------------------------------------


class TranscriptReader:
    """The one funnel through which transcripts are opened, and its ledger.

    Every open is attributed to a sealed split *before* the file is touched, so
    ``holdout_sessions_read`` in the report is a count of opens that happened
    rather than a claim about opens that did not.
    """

    def __init__(self, corpus: SealedCorpus, split: str) -> None:
        self.corpus = corpus
        self.split = split
        self.opened: Counter[str] = Counter()
        self.refused: Counter[str] = Counter()

    def load(self, entry: SessionEntry) -> SessionRecord:
        sealed_split = self.corpus.split_of(entry.session_key) or "unsealed"
        if sealed_split != self.split:
            self.refused[sealed_split] += 1
            raise EvalError(
                f"refusing to open {entry.session_key!r}: sealed split is "
                f"{sealed_split!r}, this run reads {self.split!r} only"
            )
        self.opened[sealed_split] += 1
        return load_transcript(ref_for(entry), sha256=entry.sha256)

    def accounting(self) -> dict[str, Any]:
        return {
            "split_under_measurement": self.split,
            "sessions_read_by_sealed_split": {
                name: int(self.opened.get(name, 0)) for name in (*SPLITS, "unsealed")
            },
            "opens_refused_by_sealed_split": {
                name: int(self.refused.get(name, 0)) for name in (*SPLITS, "unsealed")
            },
        }


# ---------------------------------------------------------------------------
# what one session contributes
# ---------------------------------------------------------------------------


@dataclass
class PreparedEvent:
    """One recall event as the transcript recorded it, ready to be replayed."""

    source_event_id: str
    query: str
    scope: str
    depth: Any
    max_results: int
    ambient_context: dict[str, Any]
    #: One payload per delivered node, in rank order, truncated ones included so
    #: the surviving ranks keep their live positions.
    results: list[dict[str, Any]]
    #: Node ids the scratch database will actually hold for this event.
    gradable_node_ids: tuple[str, ...]
    #: Assigned when the event is replayed into the scratch store.
    scratch_event_id: str = ""


@dataclass
class PreparedSession:
    """One eval session: its evidence, its events, and the nodes they name."""

    session_key: str
    source: str
    split_key: str
    transcript_sha256: str
    bundle: EvidenceBundle
    events: list[PreparedEvent]
    #: node id -> (content, scope, level), the nodes to reconstruct.
    nodes: dict[str, tuple[str, str, str]]


def _node_level(raw: str | None) -> str:
    value = (raw or "").strip()
    return value if value in NODE_LEVELS else "trace"


def _result_payload(node: Any, rank: int) -> dict[str, Any]:
    """The recall-result summary the live path records, rebuilt from a delivery."""

    return {
        "rank": rank + 1,
        "node_id": node.node_id,
        "level": node.level,
        "scope": node.scope,
        "score": node.score,
        "bm25_score": node.bm25_score or 0.0,
        "vector_score": node.vector_score or 0.0,
        "graph_score": node.graph_score or 0.0,
        "trigger_score": node.trigger_score or 0.0,
        "methods": list(node.methods),
    }


def prepare_session(
    record: SessionRecord, entry: SessionEntry, counters: Counter[str], **caps: int
) -> PreparedSession | None:
    """Everything one session contributes, or ``None`` when it contributes none.

    A session is usable when it recorded at least one recall event whose id the
    server handed back, that event delivered at least one node whose full content
    the transcript kept, and the session produced at least one evidence item.
    """

    nodes: dict[str, tuple[str, str, str]] = {}
    events: list[PreparedEvent] = []
    for recall in record.recalls:
        counters["recalls_seen"] += 1
        if not recall.recall_event_id:
            counters["recalls_without_event_id"] += 1
            continue
        payloads: list[dict[str, Any]] = []
        gradable: list[str] = []
        for rank, delivered in enumerate(sorted(recall.delivered, key=lambda d: d.rank)):
            payloads.append(_result_payload(delivered, rank))
            counters["deliveries_seen"] += 1
            if delivered.content_truncated:
                counters["truncated_deliveries_excluded"] += 1
                continue
            if not delivered.content:
                counters["contentless_deliveries_excluded"] += 1
                continue
            counters["deliveries_gradable"] += 1
            gradable.append(delivered.node_id)
            nodes.setdefault(
                delivered.node_id,
                (
                    delivered.content,
                    (delivered.scope or recall.scope or "global"),
                    _node_level(delivered.level),
                ),
            )
        if not gradable:
            counters["events_without_gradable_node"] += 1
            continue
        events.append(
            PreparedEvent(
                source_event_id=recall.recall_event_id,
                query=recall.query or "",
                scope=recall.scope or "global",
                depth=recall.depth,
                max_results=int(recall.max_results or len(payloads) or 10),
                ambient_context=dict(recall.ambient_context or {}),
                results=payloads,
                gradable_node_ids=tuple(gradable),
            )
        )
    if not events:
        return None
    bundle = assemble_evidence(record, **caps)
    if not bundle:
        counters["sessions_without_evidence"] += 1
        return None
    return PreparedSession(
        session_key=record.session_key,
        source=record.source,
        split_key=entry.split_key or entry.session_key,
        transcript_sha256=record.transcript_sha256 or entry.sha256,
        bundle=bundle,
        events=events,
        nodes=nodes,
    )


# ---------------------------------------------------------------------------
# the scratch database
# ---------------------------------------------------------------------------


def build_scratch_db(db_path: Path, sessions: Sequence[PreparedSession]) -> dict[str, Any]:
    """Reconstruct nodes and recall events into a fresh SQLite file.

    Nodes are inserted with the id the transcript recorded, so an event's result
    payloads resolve exactly as they did live. Written and closed before the
    server is started: one process opens this file at a time.
    """

    if db_path.exists():
        raise EvalError(f"scratch database already exists: {db_path}")
    _refuse_live_db(db_path)
    inserted = 0
    events = 0
    store = MemoryStore(db_path)
    try:
        seen: set[str] = set()
        for session in sessions:
            for node_id, (content, scope, level) in session.nodes.items():
                if node_id in seen:
                    continue
                seen.add(node_id)
                store.create_node(
                    level=level,  # type: ignore[arg-type]
                    content=content,
                    context={"scope": scope, "agent": "attest_eval"},
                    node_id=node_id,
                )
                inserted += 1
            for event in session.events:
                replayed = store.record_recall_event(
                    query=event.query,
                    scope=event.scope,
                    requested_scope=event.scope,
                    ambient_context=event.ambient_context,
                    depth=event.depth,
                    max_results=event.max_results,
                    results=event.results,
                )
                event.scratch_event_id = replayed.id
                events += 1
    finally:
        store.close()
    return {"nodes": inserted, "recall_events": events, "path": str(db_path)}


def _refuse_live_db(path: Path) -> None:
    """The one file this process must never hand to ``MemoryStore``.

    ``MemoryStore.__init__`` migrates and writes whatever it opens, and the live
    server holds that file, so opening it offline is a schema write behind the
    server's back — not a read.
    """

    try:
        same = path.resolve() == LIVE_DB.resolve()
    except OSError:  # pragma: no cover - unresolvable path is not the live one
        same = False
    if same:
        raise EvalError(f"refusing to open the live Living Memory database at {LIVE_DB}")


# ---------------------------------------------------------------------------
# the throwaway server
# ---------------------------------------------------------------------------


@dataclass
class Throwaway:
    process: subprocess.Popen[bytes]
    port: int
    token: str
    log_path: Path

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp/"

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=15)
        if self.process.poll() is None:  # pragma: no cover - stubborn child
            self.process.kill()
            self.process.wait(timeout=15)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    if port == LIVE_PORT:  # pragma: no cover - the kernel would have to hand it out
        raise EvalError("refusing to bind the live server's port")
    return port


def start_throwaway(db_path: Path, log_dir: Path, port: int | None = None) -> Throwaway:
    """Start ``python -m living_memory.server`` from this worktree over the scratch DB."""

    port = port or _free_port()
    if port == LIVE_PORT:
        raise EvalError(f"port {port} is the live unit's; pick another")
    token = secrets.token_urlsafe(32)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC_DIR), *(p for p in (env.get("PYTHONPATH"),) if p)]
    )
    env["LM_AUTH_TOKEN"] = token
    # Deterministic and offline: the measurement must not depend on a model
    # download, and nothing here reads a vector anyway.
    env.setdefault("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    log_path = log_dir / "server.log"
    handle = open(log_path, "wb")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "living_memory.server",
            "--db",
            str(db_path),
            "--transport",
            "http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--default-scope",
            "global",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    server = Throwaway(process=process, port=port, token=token, log_path=log_path)
    _await_port(server)
    return server


def _await_port(server: Throwaway) -> None:
    deadline = time.monotonic() + SERVER_BOOT_TIMEOUT
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            tail = server.log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise EvalError(f"throwaway server exited during boot:\n{tail}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex(("127.0.0.1", server.port)) == 0:
                return
        time.sleep(0.25)
    raise EvalError(f"throwaway server did not open port {server.port} in time")


# ---------------------------------------------------------------------------
# driving memory_attest over MCP
# ---------------------------------------------------------------------------


@dataclass
class Pair:
    """One (evidence, recall_event) grading."""

    arm: str
    event_session: str
    evidence_session: str
    scratch_event_id: str
    evidence: tuple[str, ...]
    evidence_sha256: str
    #: Filled from the server's verdict.
    grounded_node_ids: tuple[str, ...] = ()
    containments: tuple[float, ...] = ()
    graded_results: int = 0
    replay: bool = False
    credited: bool = False
    error: str = ""

    @property
    def grounded(self) -> bool:
        return bool(self.grounded_node_ids)

    @property
    def max_containment(self) -> float:
        return max(self.containments, default=0.0)


def _structured(result: Any) -> dict[str, Any]:
    payload = getattr(result, "structured_content", None)
    if not isinstance(payload, dict):
        raise EvalError(f"memory_attest returned no structured content: {result!r}")
    if set(payload) == {"result"} and isinstance(payload["result"], dict):
        return payload["result"]
    return payload


async def _drive(server: Throwaway, pairs: Sequence[Pair], progress_every: int) -> None:
    try:
        from fastmcp import Client
    except ImportError as error:  # pragma: no cover - environment without fastmcp
        raise EvalError(f"fastmcp is required to drive the throwaway server: {error}")

    client = Client(server.url, auth=server.token, timeout=120.0, init_timeout=60.0)
    async with client:
        served = {tool.name for tool in await client.list_tools()}
        if "memory_attest" not in served:
            raise EvalError(
                "the throwaway server does not serve memory_attest; it is not this "
                f"checkout (tools: {sorted(served)})"
            )
        for index, pair in enumerate(pairs, start=1):
            try:
                verdict = _structured(
                    await client.call_tool(
                        "memory_attest",
                        {
                            "recall_event_id": pair.scratch_event_id,
                            "evidence": list(pair.evidence),
                            "context": {
                                "agent": "attest_eval",
                                "task": f"attestation-field-check/{pair.arm}",
                                "source_session_key": pair.evidence_session,
                            },
                        },
                    )
                )
            except Exception as error:  # noqa: BLE001 - a failed grading is a finding
                pair.error = f"{type(error).__name__}: {error}"
            else:
                results = verdict.get("results") or []
                pair.grounded_node_ids = tuple(verdict.get("grounded_node_ids") or ())
                pair.containments = tuple(
                    float(item.get("containment", 0.0)) for item in results
                )
                pair.graded_results = len(results)
                pair.replay = bool(verdict.get("replay"))
                pair.credited = bool(verdict.get("credited"))
                if verdict.get("evidence_sha256") != pair.evidence_sha256:
                    raise EvalError(
                        "server recomputed a different evidence digest than the "
                        "assembler: "
                        f"{verdict.get('evidence_sha256')} != {pair.evidence_sha256}"
                    )
            if progress_every and index % progress_every == 0:
                print(f"  ... {index}/{len(pairs)} attestations", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# pairing
# ---------------------------------------------------------------------------


def derangement(sessions: Sequence[PreparedSession]) -> tuple[int, list[int]]:
    """A fixed cyclic derangement of the session list, seeded from its own keys.

    ``offset`` is drawn from ``sha256`` over the sorted session keys, so the run
    is reproducible byte-for-byte without touching ``random``. A cyclic shift by
    any offset in ``[1, n-1]`` maps no index to itself; the search then walks
    offsets deterministically until it also maps no session onto a member of its
    own identity group, because twin recordings of one conversation share a
    ``split_key`` and pairing them would be a same-session grading wearing a
    different name.
    """

    count = len(sessions)
    if count < 2:
        raise EvalError("a shuffled arm needs at least two sessions")
    seed = hashlib.sha256(
        "".join(f"{session.session_key}\n" for session in sessions).encode("utf-8")
    ).hexdigest()
    start = int(seed, 16) % (count - 1)
    groups = [session.split_key for session in sessions]
    for step in range(count - 1):
        offset = 1 + (start + step) % (count - 1)
        if all(groups[i] != groups[(i + offset) % count] for i in range(count)):
            return offset, [(i + offset) % count for i in range(count)]
    raise EvalError(
        "no cyclic offset avoids pairing a session with a twin recording of itself"
    )


def build_pairs(sessions: Sequence[PreparedSession], mapping: Sequence[int]) -> list[Pair]:
    pairs: list[Pair] = []
    for index, session in enumerate(sessions):
        other = sessions[mapping[index]]
        for event in session.events:
            pairs.append(
                Pair(
                    arm="eval",
                    event_session=session.session_key,
                    evidence_session=session.session_key,
                    scratch_event_id=event.scratch_event_id,
                    evidence=session.bundle.texts,
                    evidence_sha256=session.bundle.evidence_sha256,
                )
            )
            pairs.append(
                Pair(
                    arm="shuffled",
                    event_session=session.session_key,
                    evidence_session=other.session_key,
                    scratch_event_id=event.scratch_event_id,
                    evidence=other.bundle.texts,
                    evidence_sha256=other.bundle.evidence_sha256,
                )
            )
    return pairs


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def _histogram(values: Iterable[float]) -> dict[str, int]:
    counts = {f"[{lo:g},{hi:g})": 0 for lo, hi in zip(HISTOGRAM_EDGES, HISTOGRAM_EDGES[1:])}
    counts["[1,inf)"] = 0
    for value in values:
        placed = False
        for lo, hi in zip(HISTOGRAM_EDGES, HISTOGRAM_EDGES[1:]):
            if lo <= value < hi:
                counts[f"[{lo:g},{hi:g})"] += 1
                placed = True
                break
        if not placed:
            counts["[1,inf)"] += 1
    return counts


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return round(ordered[index], 6)

    return {
        "min": round(ordered[0], 6),
        "p50": at(0.50),
        "p90": at(0.90),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": round(ordered[-1], 6),
        "mean": round(sum(ordered) / len(ordered), 6),
    }


def summarize(arm: str, pairs: Sequence[Pair]) -> dict[str, Any]:
    graded = [pair for pair in pairs if not pair.error]
    grounded = [pair for pair in graded if pair.grounded]
    per_result = [value for pair in graded for value in pair.containments]
    per_pair_max = [pair.max_containment for pair in graded]
    sessions = {pair.event_session for pair in pairs}
    return {
        "arm": arm,
        "sessions": len(sessions),
        "pairs": len(graded),
        "pairs_attempted": len(pairs),
        "pairs_failed": len(pairs) - len(graded),
        "grounded_pairs": len(grounded),
        "grounded_rate": round(len(grounded) / len(graded), 6) if graded else 0.0,
        "sessions_with_grounded_event": len({pair.event_session for pair in grounded}),
        "grounded_node_ids": sum(len(pair.grounded_node_ids) for pair in graded),
        "graded_results": sum(pair.graded_results for pair in graded),
        "credited_pairs": sum(1 for pair in graded if pair.credited),
        "replayed_pairs": sum(1 for pair in graded if pair.replay),
        "containment": {
            "per_result_histogram": _histogram(per_result),
            "per_result": _quantiles(per_result),
            "per_pair_max_histogram": _histogram(per_pair_max),
            "per_pair_max": _quantiles(per_pair_max),
        },
        "errors": sorted({pair.error for pair in pairs if pair.error})[:10],
    }


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


#: Files whose bytes decide the measurement. Digested into the report so the
#: code that produced a number is identified even when it is not yet committed —
#: a bare ``HEAD`` would name the parent commit of a modified working tree and
#: quietly attribute the result to code that never ran.
MEASURED_SOURCES: tuple[str, ...] = (
    "src/living_memory/attestation.py",
    "src/living_memory/grounding.py",
    "src/living_memory/postsession/evidence.py",
    "scripts/attest_eval.py",
)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return ""


def _code_provenance() -> dict[str, Any]:
    digests: dict[str, str] = {}
    for name in MEASURED_SOURCES:
        path = REPO_ROOT / name
        digests[name] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
        )
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
        "measured_source_sha256": digests,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    corpus = recover_sealed_corpus(Path(args.manifest), Path(args.index))
    reader = TranscriptReader(corpus, args.split)
    caps = {
        "max_items": args.max_items,
        "max_item_chars": args.max_item_chars,
        "max_total_chars": args.max_total_chars,
    }

    counters: Counter[str] = Counter()
    keys = sorted(corpus.membership[args.split])
    prepared: list[PreparedSession] = []
    started = time.monotonic()
    for session_key in keys:
        entry = corpus.rows.get(session_key)
        if entry is None:
            counters["sealed_rows_missing_from_index"] += 1
            continue
        counters["sessions_considered"] += 1
        try:
            record = reader.load(entry)
        except EvalError:
            raise
        except Exception as error:  # noqa: BLE001 - an unparsable transcript is a finding
            counters["sessions_unparsable"] += 1
            counters[f"parse_error:{type(error).__name__}"] += 1
            continue
        session = prepare_session(record, entry, counters, **caps)
        if session is None:
            counters["sessions_unusable"] += 1
            continue
        prepared.append(session)
        if args.limit and len(prepared) >= args.limit:
            break
    load_seconds = time.monotonic() - started
    counters["sessions_usable"] = len(prepared)
    if len(prepared) < 2:
        raise EvalError(f"only {len(prepared)} usable sessions; nothing to measure")

    offset, mapping = derangement(prepared)
    scratch = Path(tempfile.mkdtemp(prefix="attest-eval-"))
    server: Throwaway | None = None
    try:
        db_path = scratch / "scratch.sqlite3"
        db_stats = build_scratch_db(db_path, prepared)
        pairs = build_pairs(prepared, mapping)
        server = start_throwaway(db_path, scratch, port=args.port)
        attest_started = time.monotonic()
        asyncio.run(_drive(server, pairs, args.progress_every))
        attest_seconds = time.monotonic() - attest_started
    finally:
        if server is not None:
            server.stop()
        if args.keep_scratch:
            print(f"scratch kept at {scratch}", file=sys.stderr)
        else:
            shutil.rmtree(scratch, ignore_errors=True)

    by_arm: dict[str, list[Pair]] = {"eval": [], "shuffled": []}
    for pair in pairs:
        by_arm[pair.arm].append(pair)

    accounting = reader.accounting()
    holdout_read = accounting["sessions_read_by_sealed_split"]["holdout"]
    eval_summary = summarize("eval", by_arm["eval"])
    shuffled_summary = summarize("shuffled", by_arm["shuffled"])
    bar = {
        "shuffled_grounded_rate_max": 0.05,
        "eval_over_shuffled_min": 3.0,
        "sessions_min": 30,
        "sessions_with_grounded_event_min": 3,
        "holdout_sessions_read_max": 0,
    }
    ratio = (
        eval_summary["grounded_rate"] / shuffled_summary["grounded_rate"]
        if shuffled_summary["grounded_rate"] > 0
        else float("inf")
    )
    passed = (
        shuffled_summary["grounded_rate"] <= bar["shuffled_grounded_rate_max"]
        and eval_summary["grounded_rate"]
        >= bar["eval_over_shuffled_min"] * shuffled_summary["grounded_rate"]
        and eval_summary["sessions"] >= bar["sessions_min"]
        and eval_summary["sessions_with_grounded_event"]
        >= bar["sessions_with_grounded_event_min"]
        and holdout_read == bar["holdout_sessions_read_max"]
    )
    provenance = _code_provenance()
    return {
        "generated_at": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "generator": "scripts/attest_eval.py",
        "commit": provenance["commit"],
        "code": provenance,
        "corpus": corpus.report(),
        "split_accounting": accounting,
        "holdout_sessions_read": holdout_read,
        "bounds": {
            "min_containment": ATTESTATION_MIN_CONTAINMENT,
            "server_max_items": EVIDENCE_MAX_ITEMS,
            "server_max_item_chars": EVIDENCE_MAX_ITEM_CHARS,
            "server_max_total_chars": EVIDENCE_MAX_TOTAL_CHARS,
            "run_caps": caps,
        },
        "scratch": {
            "database": "temporary file, removed after the run"
            if not args.keep_scratch
            else db_stats["path"],
            "nodes": db_stats["nodes"],
            "recall_events": db_stats["recall_events"],
            "server_port": server.port if server else None,
            "live_database_opened": False,
            "live_port_contacted": False,
        },
        "corpus_recovery": {
            "sessions_considered": int(counters["sessions_considered"]),
            "sessions_unparsable": int(counters["sessions_unparsable"]),
            "sessions_unusable": int(counters["sessions_unusable"]),
            "sessions_usable": int(counters["sessions_usable"]),
            "sessions_without_evidence": int(counters["sessions_without_evidence"]),
            "recalls_seen": int(counters["recalls_seen"]),
            "recalls_without_event_id": int(counters["recalls_without_event_id"]),
            "events_without_gradable_node": int(counters["events_without_gradable_node"]),
            "deliveries_seen": int(counters["deliveries_seen"]),
            "deliveries_gradable": int(counters["deliveries_gradable"]),
            "contentless_deliveries_excluded": int(
                counters["contentless_deliveries_excluded"]
            ),
            "load_seconds": round(load_seconds, 2),
            "attest_seconds": round(attest_seconds, 2),
        },
        "truncated_deliveries_excluded": int(counters["truncated_deliveries_excluded"]),
        "pairing": {
            "rule": "cyclic shift over sessions sorted by session_key",
            "seed": "sha256 over the sorted session keys, no random module",
            "offset": offset,
            "sessions": len(prepared),
            # Counted from the mapping actually used, not asserted by the rule.
            "same_group_pairs": sum(
                1
                for i, j in enumerate(mapping)
                if prepared[i].split_key == prepared[j].split_key
            ),
            "self_pairs": sum(1 for i, j in enumerate(mapping) if i == j),
            "by_source": dict(sorted(Counter(s.source for s in prepared).items())),
        },
        "eval": eval_summary,
        "shuffled": shuffled_summary,
        "bar": bar,
        "bar_result": {
            "eval_over_shuffled": None if ratio == float("inf") else round(ratio, 4),
            "shuffled_rate_is_zero": shuffled_summary["grounded_rate"] == 0.0,
            "passed": passed,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--index",
        default=str(DEFAULT_WORK_INDEX),
        help="rebuilt session index; kept out of the seal's own slot "
        f"({DEFAULT_INDEX}) — see DEFAULT_WORK_INDEX",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--split",
        default="eval",
        choices=("train", "eval"),
        help="sealed split to measure; holdout is not offered",
    )
    parser.add_argument("--limit", type=int, default=0, help="stop after N usable sessions")
    parser.add_argument("--port", type=int, default=0, help="throwaway server port (0: pick free)")
    parser.add_argument("--max-items", type=int, default=EVIDENCE_MAX_ITEMS)
    parser.add_argument("--max-item-chars", type=int, default=EVIDENCE_MAX_ITEM_CHARS)
    parser.add_argument("--max-total-chars", type=int, default=EVIDENCE_MAX_TOTAL_CHARS)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--keep-scratch", action="store_true")
    args = parser.parse_args(argv)

    if args.port == LIVE_PORT:
        print("refusing to use the live unit's port 8765", file=sys.stderr)
        return 2
    try:
        report = run(args)
    except EvalError as error:
        print(f"attest_eval: {error}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")
    print(
        "  eval     sessions={sessions} pairs={pairs} grounded={grounded_pairs} "
        "rate={grounded_rate} sessions_with_grounded={sessions_with_grounded_event}".format(
            **report["eval"]
        )
    )
    print(
        "  shuffled sessions={sessions} pairs={pairs} grounded={grounded_pairs} "
        "rate={grounded_rate}".format(**report["shuffled"])
    )
    print(f"  holdout_sessions_read={report['holdout_sessions_read']}")
    print(f"  truncated_deliveries_excluded={report['truncated_deliveries_excluded']}")
    print("BAR PASSED" if report["bar_result"]["passed"] else "BAR FAILED")
    return 0 if report["bar_result"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
