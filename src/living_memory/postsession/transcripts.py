"""Streaming transcript adapters for the post-session extraction stage.

One adapter per recording format. Every adapter obeys two rules:

**Streaming.** No adapter ever reads a transcript into memory. JSONL sources are
consumed line by line; the single-document deepseek session is consumed with an
incremental array decoder; ``result.md`` (up to 102 MB of raw CLI stdout) is
consumed line by line with a bounded tail ring buffer. Per-session state that
grows with the transcript is capped (see ``PENDING_CALL_LIMIT`` and
``MAX_BLOB_CHARS``).

**Exactness.** Identifiers are *recovered*, never re-derived. The live server's
own response is what lands in the transcript, so a recall is linked to its
``recall_event_id`` and to the exact node ids it delivered, and a write to the
node id it created -- no fuzzy matching of prose anywhere.

Where each host puts the untruncated server response:

``claude``
    a ``user`` record's ``mcpMeta.structuredContent``. The same payload is
    triplicated into ``toolUseResult`` (sometimes a JSON string, sometimes an
    object) and ``message.content[].content[].text``; ``mcpMeta`` is read first
    and the copies only serve as fallbacks for records that predate it. Results
    over ~100 KB are spilled to
    ``<project>/<session>/tool-results/*.txt`` with a ``<persisted-output>``
    stub in the transcript -- those are followed when the file still exists.
``codex``
    ``payload.type == "mcp_tool_call_end"`` with ``invocation.arguments``
    (including ``ambient_context``, which the Claude host does not send) and a
    Rust ``{"Ok": ...}`` / ``{"Err": ...}`` enum whose ``Ok.structuredContent``
    is the response.
``ae_chat``
    only truncated ``input_summary`` / ``output_summary`` strings. The
    ``recall_event_id`` sits early enough in the payload to survive truncation
    (~95% of the corpus), but delivered nodes generally do not -- full fidelity
    comes from joining ``meta.json``'s ``sessions_by_provider`` to the claude or
    codex transcript, which :mod:`living_memory.postsession.corpus` does through
    ``linked_session_ids``.
``ae_node_result``
    raw CLI stdout. Its ``session id: <uuid>`` header is the join key to the
    codex/claude JSONL; the body itself is kept only as a bounded tail.

File mutations come from Claude ``Edit``/``Write``/``MultiEdit``
(``structuredPatch`` hunk arrays plus ``originalFile``) and from codex
``patch_apply_end`` (``changes[path].{type,unified_diff|content}``).
"""

from __future__ import annotations

import ast
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Iterator

from .session import (
    MEMORY_SERVER,
    RECALL_TOOL,
    REMEMBER_TOOL,
    TEACH_TOOL,
    DeliveredNode,
    FileMutation,
    RecallInteraction,
    SessionRecord,
    SourceSpan,
    ToolCall,
    Turn,
    WriteInteraction,
    normalize_tool_name,
)

#: Largest blob (pre-image, post-image, diff, result text) kept per item.
MAX_BLOB_CHARS = 200_000

#: Largest number of in-flight tool calls tracked while streaming. Sessions
#: never have more than a handful open at once; the cap keeps a corrupted or
#: adversarial transcript from growing adapter state without bound.
PENDING_CALL_LIMIT = 256

#: Trailing lines of a ``result.md`` kept as the node's answer.
RESULT_TAIL_LINES = 400

#: Tools whose call arguments are worth holding until their result arrives.
_INTERESTING_CLAUDE_TOOLS = frozenset(
    {"Edit", "Write", "MultiEdit", "NotebookEdit", "Bash", "BashOutput"}
)

_ULID_RE = re.compile(r"[0-9A-HJKMNP-TV-Z]{26}")
_RECALL_EVENT_RE = re.compile(r'recall_event_id\\*"\s*:\s*\\*"([0-9A-HJKMNP-TV-Z]{26})')
#: Both hosts spill oversized tool output to a sidecar file and leave a stub.
#: Claude: "Full output saved to: <path>"; gigacode: "The full output has been
#: saved to: <path>".
_PERSISTED_RE = re.compile(r"full output (?:has been )?saved to:\s*(\S+)", re.IGNORECASE)
_HEADER_FIELD_RE = re.compile(r"^\s*([a-z ]+):\s*(.+?)\s*$")


# --------------------------------------------------------------------------
# streaming primitives
# --------------------------------------------------------------------------


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1 << 20) -> str:
    """Hash a file without loading it."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(record_index, record)`` for every decodable JSONL line.

    Blank lines and undecodable lines are skipped but still consume an index,
    so ``record_index`` always equals the 0-based physical line number.
    """

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except ValueError:
                continue
            if isinstance(record, dict):
                yield index, record


def iter_text_lines(path: str | os.PathLike[str]) -> Iterator[tuple[int, str]]:
    """Yield ``(line_index, line)`` for a plain-text file, streaming."""

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            yield index, line.rstrip("\n")


def _find_member(buffer: str, key: str, start: int = 0) -> int:
    """Index just past ``"key"`` used as an object member name, else ``-1``."""

    token = f'"{key}"'
    at = buffer.find(token, start)
    while at != -1:
        before = buffer.rfind("{", 0, at), buffer.rfind(",", 0, at)
        cut = max(before)
        if cut != -1 and not buffer[cut + 1 : at].strip():
            return at + len(token)
        at = buffer.find(token, at + 1)
    return -1


class _JsonMemberScanner:
    """Incremental reader for one member of a top-level JSON object.

    Deepseek stores a whole session as a single JSON document, so line
    streaming is meaningless there. This walks the file in chunks and hands
    back array elements one at a time, dropping consumed prefix as it goes, so
    peak memory is one chunk plus the largest single element.
    """

    def __init__(self, path: str | os.PathLike[str], *, chunk_size: int = 1 << 16):
        self._path = path
        self._chunk_size = chunk_size
        self._handle = None
        self._buffer = ""
        self._eof = False
        self._decoder = json.JSONDecoder()

    def __enter__(self) -> "_JsonMemberScanner":
        self._handle = open(self._path, "r", encoding="utf-8", errors="replace")
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _fill(self) -> bool:
        if self._eof or self._handle is None:
            return False
        chunk = self._handle.read(self._chunk_size)
        if not chunk:
            self._eof = True
            return False
        self._buffer += chunk
        return True

    def _seek_member(self, key: str) -> bool:
        while True:
            at = _find_member(self._buffer, key)
            if at != -1:
                self._buffer = self._buffer[at:]
                return True
            # Keep a small overlap so a member name split across chunks is found.
            if len(self._buffer) > self._chunk_size:
                self._buffer = self._buffer[-64:]
            if not self._fill():
                return False

    def _skip_to_value(self) -> bool:
        while True:
            stripped = self._buffer.lstrip()
            if stripped.startswith(":"):
                self._buffer = stripped[1:]
                return True
            if stripped and not stripped.startswith(":"):
                return False
            if not self._fill():
                return False

    def value(self, key: str) -> Any:
        """Decode the member ``key`` as a single value (used for small objects)."""

        if not self._seek_member(key) or not self._skip_to_value():
            return None
        while True:
            candidate = self._buffer.lstrip()
            try:
                value, _ = self._decoder.raw_decode(candidate)
                return value
            except ValueError:
                if not self._fill():
                    return None

    def array(self, key: str) -> Iterator[Any]:
        """Yield elements of the array member ``key`` one at a time."""

        if not self._seek_member(key) or not self._skip_to_value():
            return
        while True:
            self._buffer = self._buffer.lstrip()
            if self._buffer.startswith("["):
                self._buffer = self._buffer[1:]
                break
            if self._buffer:
                return
            if not self._fill():
                return
        while True:
            self._buffer = self._buffer.lstrip()
            while not self._buffer:
                if not self._fill():
                    return
                self._buffer = self._buffer.lstrip()
            if self._buffer[0] == ",":
                self._buffer = self._buffer[1:]
                continue
            if self._buffer[0] == "]":
                return
            while True:
                try:
                    element, end = self._decoder.raw_decode(self._buffer)
                except ValueError:
                    if not self._fill():
                        return
                    continue
                self._buffer = self._buffer[end:]
                yield element
                break


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def repo_root(cwd: str | None) -> str | None:
    """Collapse a worktree path to the repository it belongs to.

    ``/home/sfx/p/lm/.worktrees/_node_exec_x`` -> ``/home/sfx/p/lm``. Purely
    lexical: the corpus must be derivable without touching the filesystem, and
    a pruned worktree must still map to the same repo.
    """

    if not cwd:
        return None
    normalized = cwd.rstrip("/") or "/"
    marker = "/.worktrees/"
    at = normalized.find(marker)
    if at != -1:
        return normalized[:at]
    return normalized


def _clip(text: Any, limit: int = MAX_BLOB_CHARS) -> tuple[str | None, bool]:
    if text is None:
        return None, False
    value = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    if len(value) > limit:
        return value[:limit], True
    return value, False


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _first_text(blocks: Any) -> str:
    """Concatenate the ``text`` of Anthropic-style content blocks."""

    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    return "\n".join(part for part in parts if part)


def _loads(value: Any) -> Any:
    """Decode ``value`` if it is a JSON string, else return it unchanged.

    Falls back to :func:`ast.literal_eval` for hosts that serialise tool
    arguments as a Python repr (``"{'path': '...'}"``) rather than as JSON.
    ``literal_eval`` evaluates literals only -- it cannot execute anything
    from a transcript.
    """

    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            pass
        stripped = value.lstrip()
        if stripped[:1] in {"{", "["}:
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                return None
        return None
    return value


def read_spilled_json(path: str, *, max_bytes: int = 8 << 20) -> Any:
    """Read a sidecar file a host spilled an oversized tool result into."""

    candidate = Path(path)
    try:
        if not candidate.is_file() or candidate.stat().st_size > max_bytes:
            return None
        with open(candidate, "r", encoding="utf-8", errors="replace") as handle:
            return json.loads(handle.read())
    except (OSError, ValueError):
        return None


def parse_recall_payload(payload: dict[str, Any]) -> tuple[str | None, list[DeliveredNode]]:
    """Project a ``memory_recall`` server response into delivered nodes."""

    event_id = payload.get("recall_event_id")
    delivered: list[DeliveredNode] = []
    for rank, item in enumerate(payload.get("results") or []):
        if not isinstance(item, dict):
            continue
        node = item.get("node")
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if not node_id:
            continue
        content = node.get("content")
        clipped, clipped_flag = _clip(content)
        delivered.append(
            DeliveredNode(
                node_id=str(node_id),
                rank=rank,
                content=clipped,
                scope=node.get("scope"),
                level=node.get("level"),
                agent=node.get("agent"),
                task=node.get("task"),
                created_at=node.get("created_at"),
                score=_as_float(item.get("score")),
                bm25_score=_as_float(item.get("bm25_score")),
                vector_score=_as_float(item.get("vector_score")),
                graph_score=_as_float(item.get("graph_score")),
                trigger_score=_as_float(item.get("trigger_score")),
                methods=tuple(str(m) for m in item.get("methods") or ()),
                delivery=item.get("delivery"),
                content_truncated=clipped_flag or bool(item.get("content_ref")),
            )
        )
    return (str(event_id) if event_id else None), delivered


def parse_write_payload(
    payload: dict[str, Any], arguments: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Project a ``memory_remember`` / ``memory_teach`` round trip into fields.

    The server answers in two shapes and both are live in the corpus. The
    verbose one echoes the whole node (``node.content``, ``node.context``,
    ``node.provenance.prior_recalls``). The current compact one returns only
    ``node.{id,level,scope,created_at,prior_recall_count,...}`` -- there the
    written content and its context are recoverable *only* from the call
    arguments, and the recalls the write closed the loop on come from
    ``implicit_feedback.recall_event_ids`` instead of ``prior_recalls``.
    """

    arguments = arguments or {}
    node = payload.get("node") if isinstance(payload.get("node"), dict) else {}
    provenance = node.get("provenance") if isinstance(node.get("provenance"), dict) else {}
    prior_ids: list[str] = []
    prior = provenance.get("prior_recalls")
    if isinstance(prior, list):
        for entry in prior:
            if isinstance(entry, dict) and entry.get("id"):
                prior_ids.append(str(entry["id"]))
            elif isinstance(entry, str):
                prior_ids.append(entry)
    feedback = payload.get("implicit_feedback")
    feedback = feedback if isinstance(feedback, dict) else {}
    if not prior_ids:
        prior_ids = [str(item) for item in feedback.get("recall_event_ids") or ()]
    context = node.get("context") if isinstance(node.get("context"), dict) else {}
    if not context and isinstance(arguments.get("context"), dict):
        context = arguments["context"]
    written = node.get("content")
    if written is None:
        written = arguments.get("content")
    if written is None:
        written = arguments.get("correction")
    content, truncated = _clip(written)
    return {
        "node_id": str(node["id"]) if node.get("id") else None,
        "content": content,
        "context": context,
        "implicit_feedback": feedback,
        "prior_recall_ids": tuple(prior_ids),
        "transport_session_id": context.get("transport_session_id"),
        "truncated": truncated,
    }


def render_structured_patch(path: str, hunks: Any) -> str | None:
    """Rebuild unified-diff text from Claude's ``structuredPatch`` hunk arrays."""

    if not isinstance(hunks, list) or not hunks:
        return None
    lines = [f"--- a/{path}", f"+++ b/{path}"]
    for hunk in hunks:
        if not isinstance(hunk, dict):
            continue
        lines.append(
            "@@ -{0},{1} +{2},{3} @@".format(
                hunk.get("oldStart", 0),
                hunk.get("oldLines", 0),
                hunk.get("newStart", 0),
                hunk.get("newLines", 0),
            )
        )
        for line in hunk.get("lines") or []:
            lines.append(str(line))
    if len(lines) == 2:
        return None
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# roots and refs
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Roots:
    """Where the accumulated transcripts live.

    ``codex_sessions`` holds *both* codex roots: the user's own
    ``~/.codex/sessions`` and every AE-spawned
    ``<ae>/projects/<proj>/state/codex_home/sessions``. Scanning only the first
    would silently drop two thirds of the codex corpus.
    """

    claude_projects: Path | None = None
    codex_sessions: tuple[Path, ...] = ()
    gigacode_projects: Path | None = None
    deepseek_sessions: Path | None = None
    ae_projects: Path | None = None

    @classmethod
    def discover(
        cls, home: str | os.PathLike[str] | None = None, ae_root: str | os.PathLike[str] | None = None
    ) -> "Roots":
        base = Path(home) if home is not None else Path.home()
        ae = Path(ae_root) if ae_root is not None else Path("/home/sfx/p/ae")
        codex: list[Path] = [base / ".codex" / "sessions"]
        ae_projects = ae / "projects"
        if ae_projects.is_dir():
            codex.extend(
                sorted(
                    path
                    for path in ae_projects.glob("*/state/codex_home/sessions")
                    if path.is_dir()
                )
            )
        return cls(
            claude_projects=base / ".claude" / "projects",
            codex_sessions=tuple(codex),
            gigacode_projects=base / ".gigacode" / "projects",
            deepseek_sessions=base / ".deepseek" / "sessions",
            ae_projects=ae_projects,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claude_projects": str(self.claude_projects) if self.claude_projects else None,
            "codex_sessions": [str(path) for path in self.codex_sessions],
            "gigacode_projects": str(self.gigacode_projects) if self.gigacode_projects else None,
            "deepseek_sessions": str(self.deepseek_sessions) if self.deepseek_sessions else None,
            "ae_projects": str(self.ae_projects) if self.ae_projects else None,
        }


@dataclass(frozen=True, slots=True)
class TranscriptRef:
    """A discovered transcript, before it is parsed.

    ``linked_session_ids`` names the *other* transcripts that record the same
    underlying conversation (an AE chat and the claude session that served it,
    a node ``result.md`` and the codex rollout it was printed from, a subagent
    file and its parent session). The corpus unions those links so a sealed
    holdout session cannot be read through a train-labelled duplicate.
    """

    source: str
    cli: str
    path: Path
    session_key: str
    cli_session_id: str | None = None
    linked_session_ids: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------


class TranscriptAdapter:
    """Base class: discover transcripts of one format and parse them."""

    source: str = ""
    cli: str = "unknown"

    def discover(self, roots: Roots) -> Iterator[TranscriptRef]:  # pragma: no cover
        raise NotImplementedError

    def load(self, ref: TranscriptRef, *, sha256: str | None = None) -> SessionRecord:  # pragma: no cover
        raise NotImplementedError

    # -- shared record assembly -------------------------------------------

    def _new_record(self, ref: TranscriptRef, sha256: str | None) -> SessionRecord:
        return SessionRecord(
            session_key=ref.session_key,
            source=self.source,
            cli=ref.cli,
            path=str(ref.path),
            cli_session_id=ref.cli_session_id,
            linked_session_ids=tuple(sorted(set(ref.linked_session_ids))),
            transcript_sha256=sha256,
        )

    @staticmethod
    def _finalize(record: SessionRecord) -> SessionRecord:
        """Fill fractional positions once the transcript length is known.

        ``position`` is ``record_index / max(1, last_record_index)`` -- the
        share of the transcript already consumed when the interaction
        happened. That is the quantity the field measurement reports (recall
        median 0.02, remember median 0.96).
        """

        span = max(1, record.record_count - 1)
        for interaction in (*record.recalls, *record.writes):
            if interaction.record_index >= 0:
                interaction.position = min(1.0, interaction.record_index / span)
        transports: set[str] = {
            w.transport_session_id for w in record.writes if w.transport_session_id
        }
        # Codex sends the transport id on the way *in* (``ambient_context``)
        # rather than getting it echoed back on the node, so both directions
        # are harvested.
        for call in record.tool_calls:
            if call.server != MEMORY_SERVER:
                continue
            for holder in (call.arguments.get("context"), call.arguments.get("ambient_context")):
                if isinstance(holder, dict) and holder.get("transport_session_id"):
                    transports.add(str(holder["transport_session_id"]))
        record.transport_session_ids = tuple(sorted(transports))
        record.repo = repo_root(record.cwd)
        return record

    @staticmethod
    def _record_memory_call(
        record: SessionRecord,
        tool: str,
        arguments: dict[str, Any],
        payload: dict[str, Any] | None,
        span: SourceSpan,
        *,
        at: str | None = None,
        turn_index: int = -1,
        record_index: int = -1,
        truncated: bool = False,
        salvaged_recall_event_id: str | None = None,
        salvaged_node_id: str | None = None,
    ) -> None:
        """Project one Living Memory round trip onto the session record.

        Every adapter routes through here so the six hosts cannot drift apart
        in what they populate -- the fields downstream children read are
        assembled once, from the call ``arguments`` plus whatever the host kept
        of the server ``payload``.

        The ``salvaged_*`` arguments exist for the AE dashboard, whose MCP
        payloads are truncated: there the ids are recovered by pattern from the
        surviving prefix instead of decoded, and ``truncated`` says so.
        """

        payload = payload or {}
        if tool == RECALL_TOOL:
            event_id, delivered = parse_recall_payload(payload)
            record.recalls.append(
                RecallInteraction(
                    ordinal=len(record.recalls),
                    query=str(arguments.get("query") or payload.get("query") or ""),
                    span=span,
                    recall_event_id=event_id or salvaged_recall_event_id,
                    scope=arguments.get("scope") or payload.get("scope"),
                    depth=arguments.get("depth"),
                    max_results=arguments.get("max_results"),
                    ambient_context=arguments.get("ambient_context") or {},
                    delivered=delivered,
                    at=at,
                    turn_index=turn_index,
                    record_index=record_index,
                    truncated=truncated,
                )
            )
        elif tool in {REMEMBER_TOOL, TEACH_TOOL}:
            fields = parse_write_payload(payload, arguments)
            record.writes.append(
                WriteInteraction(
                    ordinal=len(record.writes),
                    kind="teach" if tool == TEACH_TOOL else "remember",
                    span=span,
                    node_id=fields["node_id"] or salvaged_node_id,
                    content=fields["content"],
                    context=fields["context"],
                    implicit_feedback=fields["implicit_feedback"],
                    prior_recall_ids=fields["prior_recall_ids"],
                    supersedes=arguments.get("trace_id") if tool == TEACH_TOOL else None,
                    transport_session_id=fields["transport_session_id"],
                    at=at,
                    turn_index=turn_index,
                    record_index=record_index,
                    truncated=truncated or fields["truncated"],
                )
            )


class ClaudeAdapter(TranscriptAdapter):
    """``~/.claude/projects/<cwd-slug>/<sessionUuid>.jsonl`` and its subagents."""

    source = "claude"
    cli = "claude"

    def discover(self, roots: Roots) -> Iterator[TranscriptRef]:
        root = roots.claude_projects
        if not root or not root.is_dir():
            return
        for path in sorted(root.rglob("*.jsonl")):
            relative = path.relative_to(root)
            parts = relative.parts
            if len(parts) == 2:
                session_id = path.stem
                yield TranscriptRef(
                    source=self.source,
                    cli=self.cli,
                    path=path,
                    session_key=f"claude:{session_id}",
                    cli_session_id=session_id,
                    linked_session_ids=(session_id,),
                )
            elif len(parts) == 4 and parts[2] == "subagents":
                parent = parts[1]
                yield TranscriptRef(
                    source=self.source,
                    cli=self.cli,
                    path=path,
                    session_key=f"claude:{parent}:{path.stem}",
                    cli_session_id=parent,
                    # Subagent transcripts belong to their parent session: they
                    # must never land in a different split than the parent.
                    linked_session_ids=(parent,),
                )

    def load(self, ref: TranscriptRef, *, sha256: str | None = None) -> SessionRecord:
        record = self._new_record(ref, sha256)
        pending: dict[str, dict[str, Any]] = {}
        turn_index = 0
        count = 0
        for index, entry in iter_jsonl(ref.path):
            count = index + 1
            kind = entry.get("type")
            if entry.get("sessionId") and not record.cli_session_id:
                record.cli_session_id = str(entry["sessionId"])
            if entry.get("cwd") and not record.cwd:
                record.cwd = str(entry["cwd"])
            if entry.get("gitBranch") and not record.git_branch:
                record.git_branch = str(entry["gitBranch"])
            stamp = entry.get("timestamp")
            if stamp:
                if not record.started_at:
                    record.started_at = str(stamp)
                record.ended_at = str(stamp)
            if kind == "assistant":
                self._read_assistant(record, entry, index, pending)
                text = _first_text((entry.get("message") or {}).get("content"))
                if text:
                    record.turns.append(
                        Turn(turn_index, "assistant", text, stamp, record.span(index))
                    )
                    turn_index += 1
            elif kind == "user":
                consumed = self._read_user(record, entry, index, pending, turn_index)
                if not consumed:
                    text = _first_text((entry.get("message") or {}).get("content"))
                    if text:
                        record.turns.append(
                            Turn(turn_index, "user", text, stamp, record.span(index))
                        )
                        turn_index += 1
        record.record_count = count
        return self._finalize(record)

    def _read_assistant(
        self,
        record: SessionRecord,
        entry: dict[str, Any],
        index: int,
        pending: dict[str, dict[str, Any]],
    ) -> None:
        message = entry.get("message") or {}
        model = message.get("model")
        if model and model != "<synthetic>":
            record.model = str(model)
        for block in message.get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            server, tool = normalize_tool_name(block.get("name"))
            call_id = block.get("id")
            arguments = block.get("input") if isinstance(block.get("input"), dict) else {}
            call = ToolCall(
                ordinal=len(record.tool_calls),
                name=tool,
                server=server,
                arguments=arguments,
                span=record.span(index, "message.content.tool_use"),
                call_id=str(call_id) if call_id else None,
                at=entry.get("timestamp"),
            )
            record.tool_calls.append(call)
            if not call_id:
                continue
            if server == MEMORY_SERVER or tool in _INTERESTING_CLAUDE_TOOLS:
                if len(pending) >= PENDING_CALL_LIMIT:
                    pending.pop(next(iter(pending)))
                pending[str(call_id)] = {
                    "server": server,
                    "tool": tool,
                    "arguments": arguments,
                    "call": call,
                    "index": index,
                    "at": entry.get("timestamp"),
                }

    def _read_user(
        self,
        record: SessionRecord,
        entry: dict[str, Any],
        index: int,
        pending: dict[str, dict[str, Any]],
        turn_index: int,
    ) -> bool:
        """Handle a tool-result-bearing ``user`` record. Returns True if consumed."""

        message = entry.get("message") or {}
        blocks = message.get("content")
        if not isinstance(blocks, list):
            return False
        consumed = False
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            consumed = True
            call_id = str(block.get("tool_use_id") or "")
            info = pending.pop(call_id, None)
            if info is None:
                continue
            call: ToolCall = info["call"]
            call.ok = not block.get("is_error")
            payload, channel = self._structured_content(entry, block, record.path)
            text, truncated = _clip(
                block.get("content") if isinstance(block.get("content"), str) else None
            )
            call.result_text = text
            call.result_truncated = truncated
            server = info["server"]
            tool = info["tool"]
            if server == MEMORY_SERVER and isinstance(payload, dict):
                self._record_memory_call(
                    record,
                    tool,
                    info["arguments"],
                    payload,
                    record.span(index, channel),
                    at=info.get("at"),
                    turn_index=turn_index,
                    record_index=info["index"],
                )
            elif tool in {"Edit", "Write", "MultiEdit", "NotebookEdit"}:
                self._add_mutation(record, tool, info, entry, index)
        return consumed

    def _structured_content(
        self, entry: dict[str, Any], block: dict[str, Any], transcript_path: str
    ) -> tuple[Any, str]:
        """Recover the untruncated MCP response and say which channel gave it.

        ``mcpMeta.structuredContent`` is authoritative and is tried first. The
        remaining channels exist only because older records predate ``mcpMeta``
        or because the host spilled a large result to a sidecar file.
        """

        meta = entry.get("mcpMeta")
        if isinstance(meta, dict) and isinstance(meta.get("structuredContent"), dict):
            return meta["structuredContent"], "mcpMeta.structuredContent"
        result = entry.get("toolUseResult")
        if isinstance(result, dict) and not {"filePath", "stdout"} & set(result):
            return result, "toolUseResult"
        decoded = _loads(result)
        if isinstance(decoded, dict):
            return decoded, "toolUseResult"
        content = block.get("content")
        if isinstance(content, list):
            for nested in content:
                if isinstance(nested, dict) and nested.get("type") == "text":
                    decoded = _loads(nested.get("text"))
                    if isinstance(decoded, dict):
                        return decoded, "message.content.text"
        if isinstance(content, str):
            match = _PERSISTED_RE.search(content)
            if match:
                spilled = read_spilled_json(match.group(1))
                if isinstance(spilled, dict):
                    return spilled, "tool-results-sidecar"
            decoded = _loads(content)
            if isinstance(decoded, dict):
                return decoded, "message.content.text"
        return None, ""

    def _add_mutation(
        self,
        record: SessionRecord,
        tool: str,
        info: dict[str, Any],
        entry: dict[str, Any],
        index: int,
    ) -> None:
        result = entry.get("toolUseResult")
        if not isinstance(result, dict):
            return
        path = result.get("filePath") or info["arguments"].get("file_path")
        if not path:
            return
        kind = result.get("type")
        kind = kind if kind in {"create", "update", "delete"} else "unknown"
        diff = render_structured_patch(str(path), result.get("structuredPatch"))
        diff_clipped, diff_truncated = _clip(diff)
        pre, pre_truncated = _clip(result.get("originalFile"))
        post, post_truncated = _clip(result.get("content"))
        record.file_mutations.append(
            FileMutation(
                ordinal=len(record.file_mutations),
                path=str(path),
                kind=kind,  # type: ignore[arg-type]
                span=record.span(index, "toolUseResult"),
                unified_diff=diff_clipped,
                pre_image=pre,
                post_image=post,
                at=entry.get("timestamp"),
                tool=tool,
                truncated=diff_truncated or pre_truncated or post_truncated,
            )
        )


class CodexAdapter(TranscriptAdapter):
    """``rollout-*.jsonl`` from ``~/.codex`` and every AE ``codex_home``."""

    source = "codex"
    cli = "codex"

    @staticmethod
    def session_id_for(path: Path) -> str:
        stem = path.stem
        if stem.startswith("rollout-") and len(stem) >= 36:
            return stem[-36:]
        return stem

    def discover(self, roots: Roots) -> Iterator[TranscriptRef]:
        for root in roots.codex_sessions:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*.jsonl")):
                session_id = self.session_id_for(path)
                yield TranscriptRef(
                    source=self.source,
                    cli=self.cli,
                    path=path,
                    session_key=f"codex:{session_id}",
                    cli_session_id=session_id,
                    linked_session_ids=(session_id,),
                )

    def load(self, ref: TranscriptRef, *, sha256: str | None = None) -> SessionRecord:
        record = self._new_record(ref, sha256)
        turn_index = 0
        count = 0
        pending_patches: dict[str, int] = {}
        for index, entry in iter_jsonl(ref.path):
            count = index + 1
            stamp = entry.get("timestamp")
            if stamp:
                if not record.started_at:
                    record.started_at = str(stamp)
                record.ended_at = str(stamp)
            envelope = entry.get("type")
            payload = entry.get("payload")
            if not isinstance(payload, dict):
                continue
            if envelope == "session_meta":
                self._read_session_meta(record, payload)
                continue
            if envelope == "turn_context":
                if payload.get("model"):
                    record.model = str(payload["model"])
                if payload.get("cwd") and not record.cwd:
                    record.cwd = str(payload["cwd"])
                continue
            kind = payload.get("type")
            if kind == "mcp_tool_call_end":
                self._read_mcp_end(record, payload, index, stamp, turn_index)
            elif kind == "patch_apply_end":
                self._read_patch(record, payload, index, stamp)
            elif kind in {"custom_tool_call", "function_call"}:
                self._read_call(record, payload, index, stamp, pending_patches)
            elif kind in {"custom_tool_call_output", "function_call_output"}:
                ordinal = pending_patches.pop(str(payload.get("call_id") or ""), None)
                if ordinal is not None and ordinal < len(record.tool_calls):
                    call = record.tool_calls[ordinal]
                    text, truncated = _clip(payload.get("output"))
                    call.result_text = text
                    call.result_truncated = truncated
                    call.ok = True
            elif kind == "user_message":
                text = str(payload.get("message") or "")
                if text:
                    record.turns.append(
                        Turn(turn_index, "user", text, stamp, record.span(index))
                    )
                    turn_index += 1
            elif kind == "agent_message":
                text = str(payload.get("message") or "")
                if text:
                    record.turns.append(
                        Turn(turn_index, "assistant", text, stamp, record.span(index))
                    )
                    turn_index += 1
        record.record_count = count
        return self._finalize(record)

    @staticmethod
    def _read_session_meta(record: SessionRecord, payload: dict[str, Any]) -> None:
        if payload.get("cwd") and not record.cwd:
            record.cwd = str(payload["cwd"])
        session_id = payload.get("session_id") or payload.get("id")
        if session_id and not record.cli_session_id:
            record.cli_session_id = str(session_id)
        git = payload.get("git")
        if isinstance(git, dict):
            if git.get("branch") and not record.git_branch:
                record.git_branch = str(git["branch"])
            if git.get("commit_hash") and not record.git_commit:
                record.git_commit = str(git["commit_hash"])

    def _read_mcp_end(
        self,
        record: SessionRecord,
        payload: dict[str, Any],
        index: int,
        stamp: Any,
        turn_index: int,
    ) -> None:
        invocation = payload.get("invocation")
        if not isinstance(invocation, dict):
            return
        server = invocation.get("server")
        tool = str(invocation.get("tool") or "")
        arguments = invocation.get("arguments")
        arguments = arguments if isinstance(arguments, dict) else {}
        result = payload.get("result")
        ok_body, ok = self._unwrap_result(result)
        call = ToolCall(
            ordinal=len(record.tool_calls),
            name=tool,
            server=str(server) if server else None,
            arguments=arguments,
            span=record.span(index, "payload.invocation"),
            call_id=payload.get("call_id"),
            at=stamp,
            ok=ok,
        )
        record.tool_calls.append(call)
        if server != MEMORY_SERVER or not isinstance(ok_body, dict):
            return
        structured = ok_body.get("structuredContent")
        if not isinstance(structured, dict):
            structured = self._structured_from_content(ok_body)
        if not isinstance(structured, dict):
            return
        self._record_memory_call(
            record,
            tool,
            arguments,
            structured,
            record.span(index, "payload.result.Ok.structuredContent"),
            at=stamp,
            turn_index=turn_index,
            record_index=index,
        )

    @staticmethod
    def _unwrap_result(result: Any) -> tuple[Any, bool | None]:
        """Decode the Rust ``{"Ok": ...}`` / ``{"Err": ...}`` enum."""

        if not isinstance(result, dict):
            return None, None
        if "Ok" in result:
            body = result["Ok"]
            ok = not (isinstance(body, dict) and body.get("isError"))
            return body, ok
        if "Err" in result:
            return result["Err"], False
        return result, None

    @staticmethod
    def _structured_from_content(body: dict[str, Any]) -> Any:
        """Older codex rollouts carry only the text block; decode it."""

        for block in body.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                decoded = _loads(block.get("text"))
                if isinstance(decoded, dict):
                    return decoded
        return None

    def _read_patch(
        self, record: SessionRecord, payload: dict[str, Any], index: int, stamp: Any
    ) -> None:
        changes = payload.get("changes")
        if not isinstance(changes, dict):
            return
        for path in sorted(changes):
            change = changes[path]
            if not isinstance(change, dict):
                continue
            kind = change.get("type")
            kind = kind if kind in {"create", "update", "delete", "add"} else "unknown"
            kind = "create" if kind == "add" else kind
            diff, diff_truncated = _clip(change.get("unified_diff"))
            post, post_truncated = _clip(change.get("content"))
            record.file_mutations.append(
                FileMutation(
                    ordinal=len(record.file_mutations),
                    path=str(path),
                    kind=kind,  # type: ignore[arg-type]
                    span=record.span(index, "payload.changes"),
                    unified_diff=diff,
                    post_image=post,
                    at=stamp,
                    tool="apply_patch",
                    truncated=diff_truncated or post_truncated,
                )
            )

    def _read_call(
        self,
        record: SessionRecord,
        payload: dict[str, Any],
        index: int,
        stamp: Any,
        pending: dict[str, int],
    ) -> None:
        name = str(payload.get("name") or "")
        arguments = payload.get("arguments")
        if isinstance(arguments, str):
            arguments = _loads(arguments)
        if not isinstance(arguments, dict):
            arguments = {"input": payload.get("input")} if payload.get("input") else {}
        server, tool = normalize_tool_name(name)
        call = ToolCall(
            ordinal=len(record.tool_calls),
            name=tool or name,
            server=payload.get("namespace") or server,
            arguments=arguments,
            span=record.span(index, "payload"),
            call_id=payload.get("call_id"),
            at=stamp,
        )
        record.tool_calls.append(call)
        call_id = payload.get("call_id")
        if call_id:
            if len(pending) >= PENDING_CALL_LIMIT:
                pending.pop(next(iter(pending)))
            pending[str(call_id)] = call.ordinal


class GigacodeAdapter(TranscriptAdapter):
    """``~/.gigacode/projects/<slug>/chats/<uuid>.jsonl``.

    A Claude-shaped envelope wrapped around Gemini-shaped ``message.parts[]``
    with ``functionCall`` / ``functionResponse`` instead of tool blocks.
    """

    source = "gigacode"
    cli = "gigacode"

    def discover(self, roots: Roots) -> Iterator[TranscriptRef]:
        root = roots.gigacode_projects
        if not root or not root.is_dir():
            return
        for path in sorted(root.glob("*/chats/*.jsonl")):
            session_id = path.stem
            yield TranscriptRef(
                source=self.source,
                cli=self.cli,
                path=path,
                session_key=f"gigacode:{session_id}",
                cli_session_id=session_id,
                linked_session_ids=(session_id,),
            )

    def load(self, ref: TranscriptRef, *, sha256: str | None = None) -> SessionRecord:
        record = self._new_record(ref, sha256)
        pending: dict[str, dict[str, Any]] = {}
        turn_index = 0
        count = 0
        for index, entry in iter_jsonl(ref.path):
            count = index + 1
            stamp = entry.get("timestamp")
            if stamp:
                if not record.started_at:
                    record.started_at = str(stamp)
                record.ended_at = str(stamp)
            if entry.get("cwd") and not record.cwd:
                record.cwd = str(entry["cwd"])
            if entry.get("model"):
                record.model = str(entry["model"])
            message = entry.get("message")
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or entry.get("type") or "")
            texts: list[str] = []
            for part in message.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                if "text" in part:
                    texts.append(str(part.get("text") or ""))
                elif "functionCall" in part:
                    self._read_call(record, part["functionCall"], index, stamp, pending)
                elif "functionResponse" in part:
                    self._read_response(
                        record, part["functionResponse"], index, pending, turn_index
                    )
            text = "\n".join(part for part in texts if part)
            if text:
                normalized = "assistant" if role in {"model", "assistant"} else "user"
                record.turns.append(
                    Turn(turn_index, normalized, text, stamp, record.span(index))
                )
                turn_index += 1
        record.record_count = count
        return self._finalize(record)

    def _read_call(
        self,
        record: SessionRecord,
        payload: Any,
        index: int,
        stamp: Any,
        pending: dict[str, dict[str, Any]],
    ) -> None:
        if not isinstance(payload, dict):
            return
        server, tool = normalize_tool_name(payload.get("name"))
        arguments = payload.get("args") if isinstance(payload.get("args"), dict) else {}
        call = ToolCall(
            ordinal=len(record.tool_calls),
            name=tool,
            server=server,
            arguments=arguments,
            span=record.span(index, "message.parts.functionCall"),
            call_id=payload.get("id"),
            at=stamp,
        )
        record.tool_calls.append(call)
        call_id = payload.get("id")
        if call_id:
            if len(pending) >= PENDING_CALL_LIMIT:
                pending.pop(next(iter(pending)))
            pending[str(call_id)] = {
                "server": server,
                "tool": tool,
                "arguments": arguments,
                "call": call,
                "index": index,
                "at": stamp,
            }

    @staticmethod
    def _structured(response: dict[str, Any]) -> Any:
        """Recover the server response from gigacode's ``{"output": "<json>"}``.

        This host neither forwards ``structuredContent`` nor keeps content
        blocks: the whole MCP reply arrives as one JSON string under
        ``output``, and an oversized one is spilled to
        ``~/.gigacode/tmp/**/<tool>_<hash>.output`` with a stub in its place.
        """

        structured = response.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        structured = CodexAdapter._structured_from_content(response)
        if isinstance(structured, dict):
            return structured
        output = response.get("output")
        if isinstance(output, str):
            match = _PERSISTED_RE.search(output)
            if match:
                spilled = read_spilled_json(match.group(1))
                if isinstance(spilled, dict):
                    return spilled
            decoded = _loads(output)
            if isinstance(decoded, dict):
                return decoded
        return None

    def _read_response(
        self,
        record: SessionRecord,
        payload: Any,
        index: int,
        pending: dict[str, dict[str, Any]],
        turn_index: int,
    ) -> None:
        if not isinstance(payload, dict):
            return
        info = pending.pop(str(payload.get("id") or ""), None)
        if info is None:
            return
        response = payload.get("response")
        call: ToolCall = info["call"]
        call.ok = not (isinstance(response, dict) and response.get("error"))
        text, truncated = _clip(response)
        call.result_text = text
        call.result_truncated = truncated
        if info["server"] != MEMORY_SERVER or not isinstance(response, dict):
            return
        structured = self._structured(response)
        if not isinstance(structured, dict):
            return
        self._record_memory_call(
            record,
            info["tool"],
            info["arguments"],
            structured,
            record.span(index, "message.parts.functionResponse"),
            at=info.get("at"),
            turn_index=turn_index,
            record_index=info["index"],
        )


class DeepseekAdapter(TranscriptAdapter):
    """``~/.deepseek/sessions/<uuid>.json`` -- one JSON document per session.

    Line streaming is meaningless here, so the adapter walks the document with
    :class:`_JsonMemberScanner`, decoding ``messages`` one element at a time.

    The seven sessions accumulated as of 2026-08-19 call this host's built-in
    tools only (``edit_file``, ``grep_files``, ``task_gate_run``, ...) and reach
    Living Memory just once, through ``list_mcp_resources``. The memory branch
    below is therefore unexercised by today's corpus but not speculative: this
    host stores tool blocks in Anthropic's own shape, so a ``memory_recall``
    lands in exactly the form :class:`ClaudeAdapter` already reads, and without
    the branch its identifiers would sit in ``result_text`` and never reach the
    schema. The fixture pins that shape.
    """

    source = "deepseek"
    cli = "deepseek"

    def discover(self, roots: Roots) -> Iterator[TranscriptRef]:
        root = roots.deepseek_sessions
        if not root or not root.is_dir():
            return
        for path in sorted(root.glob("*.json")):
            session_id = path.stem
            yield TranscriptRef(
                source=self.source,
                cli=self.cli,
                path=path,
                session_key=f"deepseek:{session_id}",
                cli_session_id=session_id,
                linked_session_ids=(session_id,),
            )

    def load(self, ref: TranscriptRef, *, sha256: str | None = None) -> SessionRecord:
        record = self._new_record(ref, sha256)
        with _JsonMemberScanner(ref.path) as scanner:
            metadata = scanner.value("metadata")
        if isinstance(metadata, dict):
            record.model = metadata.get("model")
            record.cwd = metadata.get("workspace")
            record.started_at = metadata.get("created_at")
            record.ended_at = metadata.get("updated_at")
            if metadata.get("id"):
                record.cli_session_id = str(metadata["id"])
        pending: dict[str, dict[str, Any]] = {}
        turn_index = 0
        count = 0
        with _JsonMemberScanner(ref.path) as scanner:
            for index, message in enumerate(scanner.array("messages")):
                count = index + 1
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role") or "")
                blocks = message.get("content")
                if isinstance(blocks, str):
                    blocks = [{"type": "text", "text": blocks}]
                texts: list[str] = []
                for block in blocks or []:
                    if not isinstance(block, dict):
                        continue
                    kind = block.get("type")
                    if kind == "text":
                        texts.append(str(block.get("text") or ""))
                    elif kind == "tool_use":
                        self._read_call(record, block, index, pending)
                    elif kind == "tool_result":
                        self._read_result(record, block, index, pending, turn_index)
                text = "\n".join(part for part in texts if part)
                if text:
                    normalized = "assistant" if role == "assistant" else role or "user"
                    record.turns.append(
                        Turn(turn_index, normalized, text, None, record.span(index))
                    )
                    turn_index += 1
        record.record_count = count
        return self._finalize(record)

    def _read_call(
        self,
        record: SessionRecord,
        block: dict[str, Any],
        index: int,
        pending: dict[str, dict[str, Any]],
    ) -> None:
        server, tool = normalize_tool_name(block.get("name"))
        # This host records tool arguments as a Python repr string as often as
        # it records them as an object.
        arguments = block.get("input")
        if not isinstance(arguments, dict):
            arguments = _loads(arguments)
        arguments = arguments if isinstance(arguments, dict) else {}
        call = ToolCall(
            ordinal=len(record.tool_calls),
            name=tool,
            server=server,
            arguments=arguments,
            span=record.span(index, "messages.content.tool_use"),
            call_id=block.get("id"),
        )
        record.tool_calls.append(call)
        interesting = server == MEMORY_SERVER or tool in {
            "edit_file",
            "write_file",
            "apply_patch",
        }
        if interesting and block.get("id"):
            if len(pending) >= PENDING_CALL_LIMIT:
                pending.pop(next(iter(pending)))
            pending[str(block["id"])] = {
                "server": server,
                "tool": tool,
                "arguments": arguments,
                "call": call,
                "index": index,
            }

    @staticmethod
    def _structured(content: Any) -> Any:
        """Recover an MCP response from an Anthropic-shaped ``tool_result``.

        Deepseek stores tool results exactly the way the Anthropic API returns
        them, so the payload is either the JSON text of a single ``text`` block
        or a bare JSON string.
        """

        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    decoded = _loads(block.get("text"))
                    if isinstance(decoded, dict):
                        return decoded
            return None
        decoded = _loads(content)
        return decoded if isinstance(decoded, dict) else None

    def _read_result(
        self,
        record: SessionRecord,
        block: dict[str, Any],
        index: int,
        pending: dict[str, dict[str, Any]],
        turn_index: int,
    ) -> None:
        info = pending.pop(str(block.get("tool_use_id") or ""), None)
        if info is None:
            return
        call: ToolCall = info["call"]
        text, truncated = _clip(block.get("content"))
        call.result_text = text
        call.result_truncated = truncated
        call.ok = not block.get("is_error")
        if info.get("server") == MEMORY_SERVER:
            # The blocks here are Anthropic's own, so a Living Memory reply
            # arrives in the same shape the Claude host records -- without this
            # branch the recall event id and the delivered node ids would be
            # visible in ``result_text`` yet never recovered into the schema.
            structured = self._structured(block.get("content"))
            if isinstance(structured, dict):
                self._record_memory_call(
                    record,
                    info["tool"],
                    info["arguments"],
                    structured,
                    record.span(index, "messages.content.tool_result"),
                    turn_index=turn_index,
                    record_index=info["index"],
                )
            return
        arguments = info["arguments"]
        path = arguments.get("path") or arguments.get("file_path")
        if not path:
            return
        # This host spells a replacement edit ``search``/``replace`` and a
        # whole-file write ``content``.
        diff, diff_truncated = _clip(arguments.get("diff") or arguments.get("patch"))
        pre, pre_truncated = _clip(arguments.get("old_string") or arguments.get("search"))
        post, post_truncated = _clip(
            arguments.get("content")
            or arguments.get("new_string")
            or arguments.get("replace")
        )
        record.file_mutations.append(
            FileMutation(
                ordinal=len(record.file_mutations),
                path=str(path),
                kind="create" if info["tool"] == "write_file" else "update",
                span=record.span(index, "messages.content.tool_use"),
                unified_diff=diff,
                pre_image=pre,
                post_image=post,
                tool=info["tool"],
                truncated=diff_truncated or pre_truncated or post_truncated,
            )
        )


class AeChatAdapter(TranscriptAdapter):
    """``<ae>/projects/<proj>/state/chats/<chatId>/transcript.jsonl``.

    The dashboard records only truncated ``input_summary`` / ``output_summary``
    strings for MCP payloads, so recall ids are salvaged by regex from what
    survived truncation and the delivered nodes are recovered by joining the
    provider session ids from ``meta.json``.
    """

    source = "ae_chat"
    cli = "ae"

    def discover(self, roots: Roots) -> Iterator[TranscriptRef]:
        root = roots.ae_projects
        if not root or not root.is_dir():
            return
        for path in sorted(root.glob("*/state/chats/*/transcript.jsonl")):
            chat_id = path.parent.name
            linked = self._linked_ids(path.parent / "meta.json")
            yield TranscriptRef(
                source=self.source,
                cli=self.cli,
                path=path,
                session_key=f"ae_chat:{chat_id}",
                cli_session_id=chat_id,
                linked_session_ids=linked,
            )

    @staticmethod
    def _linked_ids(meta_path: Path) -> tuple[str, ...]:
        try:
            if not meta_path.is_file() or meta_path.stat().st_size > 4 << 20:
                return ()
            with open(meta_path, "r", encoding="utf-8", errors="replace") as handle:
                meta = json.load(handle)
        except (OSError, ValueError):
            return ()
        linked: set[str] = set()
        by_provider = meta.get("sessions_by_provider")
        if isinstance(by_provider, dict):
            for value in by_provider.values():
                if isinstance(value, str) and value:
                    linked.add(value)
        session = meta.get("session")
        if isinstance(session, dict) and isinstance(session.get("token"), str):
            linked.add(session["token"])
        return tuple(sorted(linked))

    def load(self, ref: TranscriptRef, *, sha256: str | None = None) -> SessionRecord:
        record = self._new_record(ref, sha256)
        meta_path = ref.path.parent / "meta.json"
        meta: dict[str, Any] = {}
        try:
            if meta_path.is_file():
                with open(meta_path, "r", encoding="utf-8", errors="replace") as handle:
                    loaded = json.load(handle)
                meta = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            record.parse_warnings.append("meta.json unreadable")
        record.model = meta.get("model")
        turn_index = 0
        count = 0
        pending: dict[str, dict[str, Any]] = {}
        turn_cli: str | None = None
        for index, entry in iter_jsonl(ref.path):
            count = index + 1
            stamp = self._stamp(entry.get("ts"))
            if stamp:
                if not record.started_at:
                    record.started_at = stamp
                record.ended_at = stamp
            kind = entry.get("type")
            if kind == "turn_start":
                turn_cli = entry.get("provider")
                if entry.get("model"):
                    record.model = str(entry["model"])
            elif kind == "user_message":
                text = str(entry.get("text") or "")
                if text:
                    record.turns.append(
                        Turn(turn_index, "user", text, stamp, record.span(index), cli=turn_cli)
                    )
                    turn_index += 1
            elif kind == "tool_use":
                self._read_call(record, entry, index, stamp, pending)
            elif kind == "tool_result":
                self._read_result(record, entry, index, pending, turn_index)
            elif kind == "raw" and entry.get("thread_id"):
                thread = str(entry["thread_id"])
                if thread not in record.linked_session_ids:
                    record.linked_session_ids = tuple(
                        sorted({*record.linked_session_ids, thread})
                    )
        record.cwd = meta.get("workspace") or record.cwd
        record.record_count = count
        return self._finalize(record)

    @staticmethod
    def _stamp(value: Any) -> str | None:
        """Dashboard timestamps are epoch milliseconds."""

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        return (
            datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    def _read_call(
        self,
        record: SessionRecord,
        entry: dict[str, Any],
        index: int,
        stamp: str | None,
        pending: dict[str, dict[str, Any]],
    ) -> None:
        server, tool = normalize_tool_name(entry.get("name"))
        summary = entry.get("input_summary")
        arguments = _loads(summary)
        truncated = not isinstance(arguments, dict)
        if not isinstance(arguments, dict):
            arguments = {"_input_summary": summary} if summary else {}
        call = ToolCall(
            ordinal=len(record.tool_calls),
            name=tool,
            server=server,
            arguments=arguments,
            span=record.span(index, "input_summary"),
            call_id=entry.get("tool_id"),
            at=stamp,
            result_truncated=truncated,
        )
        record.tool_calls.append(call)
        tool_id = entry.get("tool_id")
        if tool_id:
            if len(pending) >= PENDING_CALL_LIMIT:
                pending.pop(next(iter(pending)))
            pending[str(tool_id)] = {
                "server": server,
                "tool": tool,
                "arguments": arguments,
                "summary": summary,
                "call": call,
                "index": index,
                "at": stamp,
            }

    def _read_result(
        self,
        record: SessionRecord,
        entry: dict[str, Any],
        index: int,
        pending: dict[str, dict[str, Any]],
        turn_index: int,
    ) -> None:
        info = pending.pop(str(entry.get("tool_id") or ""), None)
        if info is None:
            return
        call: ToolCall = info["call"]
        call.ok = bool(entry.get("ok"))
        summary = entry.get("output_summary")
        text, clipped = _clip(summary)
        call.result_text = text
        call.result_truncated = True
        if info["server"] != MEMORY_SERVER:
            return
        # Nothing survives the dashboard's truncation intact, so the ids are
        # salvaged by pattern from whatever prefix it kept: the recall event id
        # sits early enough in the payload to survive ~95% of the time, the
        # written node id is the first ULID in the response.
        payload = self._decode_summary(summary)
        arguments = dict(info["arguments"])
        if not arguments.get("query"):
            arguments["query"] = self._salvage_query(info.get("summary"))
        self._record_memory_call(
            record,
            info["tool"],
            arguments,
            payload if isinstance(payload, dict) else None,
            record.span(index, "output_summary"),
            at=info.get("at"),
            turn_index=turn_index,
            record_index=info["index"],
            truncated=True,
            salvaged_recall_event_id=self._salvage_recall_event_id(summary),
            salvaged_node_id=self._salvage_node_id(summary),
        )

    @staticmethod
    def _salvage_recall_event_id(summary: Any) -> str | None:
        if not isinstance(summary, str):
            return None
        match = _RECALL_EVENT_RE.search(summary)
        return match.group(1) if match else None

    @staticmethod
    def _decode_summary(summary: Any) -> Any:
        """Decode an ``output_summary`` when truncation happened to spare it."""

        outer = _loads(summary)
        if not isinstance(outer, dict):
            return None
        for block in outer.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                inner = _loads(block.get("text"))
                if isinstance(inner, dict):
                    return inner
        return outer if "recall_event_id" in outer or "node" in outer else None

    @staticmethod
    def _salvage_query(summary: Any) -> str:
        if not isinstance(summary, str):
            return ""
        match = re.search(r'"query"\s*:\s*"((?:[^"\\]|\\.)*)', summary)
        if not match:
            return ""
        try:
            return json.loads(f'"{match.group(1)}"')
        except ValueError:
            return match.group(1)

    @staticmethod
    def _salvage_node_id(summary: Any) -> str | None:
        if not isinstance(summary, str):
            return None
        match = _ULID_RE.search(summary)
        return match.group(0) if match else None


class AeNodeResultAdapter(TranscriptAdapter):
    """``<ae>/projects/<proj>/state/goals/**/result.md`` -- raw CLI stdout.

    Files reach 102 MB, so the body is streamed and only a bounded tail is
    kept. The value of this source is its header: ``session id: <uuid>`` joins
    the node back to the codex or claude JSONL that has the real identifiers.
    """

    source = "ae_node_result"
    cli = "unknown"

    def discover(self, roots: Roots) -> Iterator[TranscriptRef]:
        root = roots.ae_projects
        if not root or not root.is_dir():
            return
        for path in sorted(root.glob("*/state/goals/**/result.md")):
            try:
                if path.stat().st_size == 0:
                    continue
            except OSError:
                continue
            header = self._read_header(path)
            session_id = header.get("session id")
            project = path.relative_to(root).parts[0]
            goal_path = self._goal_path(path, root)
            yield TranscriptRef(
                source=self.source,
                cli=header.get("_cli", "unknown"),
                path=path,
                session_key=f"ae_node:{project}:{goal_path}",
                cli_session_id=session_id,
                linked_session_ids=(session_id,) if session_id else (),
            )

    @staticmethod
    def _goal_path(path: Path, root: Path) -> str:
        parts = path.relative_to(root).parts
        # <project>/state/goals/<...>/result.md
        return "/".join(parts[3:-1]) or "root"

    #: Banner keys worth keeping; ``session id`` is the join key.
    HEADER_KEYS = frozenset(
        {"session id", "model", "provider", "workdir", "reasoning effort"}
    )

    @classmethod
    def parse_header(cls, lines: Iterable[str], limit: int = 24) -> dict[str, str]:
        """Parse the CLI banner from the first ``limit`` lines of stdout."""

        header: dict[str, str] = {}
        for count, line in enumerate(lines):
            if count >= limit:
                break
            if line.startswith("OpenAI Codex"):
                header["_cli"] = "codex"
                header["cli_version"] = line.strip()
                continue
            match = _HEADER_FIELD_RE.match(line)
            if match and match.group(1).strip() in cls.HEADER_KEYS:
                header[match.group(1).strip()] = match.group(2).strip()
        return header

    @classmethod
    def _read_header(cls, path: Path, limit: int = 24) -> dict[str, str]:
        """Parse the banner without reading past it."""

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                return cls.parse_header(handle, limit)
        except OSError:
            return {}

    def load(self, ref: TranscriptRef, *, sha256: str | None = None) -> SessionRecord:
        record = self._new_record(ref, sha256)
        header = self._read_header(ref.path)
        record.model = header.get("model")
        record.cwd = header.get("workdir")
        record.cli_session_id = header.get("session id") or ref.cli_session_id
        tail: deque[tuple[int, str]] = deque(maxlen=RESULT_TAIL_LINES)
        count = 0
        for index, line in iter_text_lines(ref.path):
            count = index + 1
            tail.append((index, line))
        record.record_count = count
        if tail:
            start = tail[0][0]
            text, truncated = _clip("\n".join(line for _, line in tail))
            record.turns.append(
                Turn(
                    0,
                    "assistant",
                    text or "",
                    None,
                    record.span(start, "tail"),
                    cli=ref.cli,
                    truncated=truncated or count > RESULT_TAIL_LINES,
                )
            )
        if not ref.linked_session_ids and record.cli_session_id:
            record.linked_session_ids = (record.cli_session_id,)
        record.parse_warnings.append(
            "ae_node_result carries no structured identifiers; join "
            "linked_session_ids to the codex/claude transcript for fidelity"
        )
        return self._finalize(record)


ADAPTERS: dict[str, TranscriptAdapter] = {
    adapter.source: adapter
    for adapter in (
        ClaudeAdapter(),
        CodexAdapter(),
        AeChatAdapter(),
        AeNodeResultAdapter(),
        GigacodeAdapter(),
        DeepseekAdapter(),
    )
}


def discover_transcripts(roots: Roots, sources: Iterable[str] | None = None) -> Iterator[TranscriptRef]:
    """Enumerate every transcript under ``roots``, source by source, sorted."""

    wanted = tuple(sources) if sources else tuple(ADAPTERS)
    for name in wanted:
        adapter = ADAPTERS.get(name)
        if adapter is None:
            raise KeyError(f"unknown transcript source: {name!r}")
        yield from adapter.discover(roots)


def load_transcript(ref: TranscriptRef, *, sha256: str | None = None) -> SessionRecord:
    """Parse one discovered transcript into a :class:`SessionRecord`."""

    adapter = ADAPTERS.get(ref.source)
    if adapter is None:
        raise KeyError(f"unknown transcript source: {ref.source!r}")
    return adapter.load(ref, sha256=sha256)
