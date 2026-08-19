"""The sealed session corpus for the post-session extraction stage.

``corpus.py`` enumerates every transcript the machine has accumulated, hashes
it, and assigns it to one of three splits. What it writes is the seal later
children must not be able to move quietly.

Two files, on purpose
---------------------
The manifest and the per-session rows are **separate artifacts**:

``artifacts/post-session/corpus.json`` (tracked, ~130 lines)
    manifest_version, generator, generated_at, roots, sources, split_policy,
    aggregate counts, the per-split seal digests, ``holdout_sha256``, and
    ``index_sha256`` binding the manifest to the index. No per-session rows.

``artifacts/post-session/corpus-index.jsonl`` (ignored, rebuildable)
    one compact JSON object per session, sorted by ``session_key``.

The first draft of this module serialized every row into the tracked manifest.
On the real corpus that is 89519 lines in one file -- a single artifact that
breaches a 4000-line merge gate on its own, for a payload that is a few
thousand absolute paths under the user's home directory and is regenerated
from disk in two minutes. The digest is the part worth tracking; the rows are
not, so they are written beside it and ignored. ``index_sha256`` is what keeps
the split: an index that does not hash to the tracked digest is not the index
this manifest sealed, and ``--verify`` says so before it trusts a single row.

Split rule
----------
::

    bucket = int(sha256(split_key).hexdigest(), 16) % 10
    0-5 -> train      6-7 -> eval      8-9 -> holdout

``split_key`` is the key of the *identity group* the transcript belongs to.
The same conversation is frequently recorded more than once:

* an AE dashboard chat and the claude/codex session that actually served it
  (``meta.json``'s ``sessions_by_provider``),
* a goal node's ``result.md`` and the codex rollout it was printed from
  (the ``session id:`` banner),
* a Claude subagent transcript and its parent session (directory layout).

Bucketing each file independently would scatter those copies across splits,
and a holdout session would then be readable through its train-labelled twin.
So transcripts are unioned by their shared session identifiers first, and the
group takes the bucket of its root -- the lexicographically smallest token in
the component, over both session keys and the raw ``id:<uuid>`` identifiers
the members declare. Including the raw identifiers is what keeps the key
stable while the Claude window rolls: when a claude transcript ages out, the
AE chat that named it still contributes ``id:<uuid>``, so the surviving
members keep the bucket they were sealed with. Every source prefix
(``ae_chat``, ``ae_node``, ``claude``, ``codex``, ``deepseek``, ``gigacode``)
sorts before ``id:``, so the root is always a real session key -- and for a
transcript with no twin, which is the overwhelming majority, ``split_key`` is
its own ``session_key`` and the rule above applies verbatim.

Seal
----
``holdout_sha256`` is ``sha256("\\n".join(sorted(holdout session keys)) +
"\\n")``. Any later child that quietly moves a session in or out of the
holdout changes that digest, and ``--verify`` says so. The holdout *keys*
never reach the tracked manifest at all -- only the digest over them.

Content policy
--------------
Manifest and index carry paths, hashes, byte and record counts, timestamps,
identifiers and split labels. They carry **no transcript content**, and
:func:`assert_no_content` enforces the field whitelist -- on the manifest and
on every index row -- rather than trusting the writer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from .session import SessionRecord
from .transcripts import (
    ADAPTERS,
    Roots,
    TranscriptRef,
    _JsonMemberScanner,
    discover_transcripts,
    load_transcript,
    repo_root,
)

#: Bumped from 1 when the per-session rows moved out of the manifest.
MANIFEST_VERSION = 2
GENERATOR = "scripts/postsession_corpus.py"
DEFAULT_MANIFEST = Path("artifacts/post-session/corpus.json")
DEFAULT_INDEX = Path("artifacts/post-session/corpus-index.jsonl")

BUCKETS = 10
TRAIN_BUCKETS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
EVAL_BUCKETS: tuple[int, ...] = (6, 7)
HOLDOUT_BUCKETS: tuple[int, ...] = (8, 9)
SPLITS: tuple[str, ...] = ("train", "eval", "holdout")

#: Every key an index row may carry. Nothing here can hold prose.
ENTRY_FIELDS: tuple[str, ...] = (
    "session_key",
    "source",
    "cli",
    "path",
    "bytes",
    "sha256",
    "records",
    "started_at",
    "ended_at",
    "cwd",
    "repo",
    "git_branch",
    "model",
    "cli_session_id",
    "linked_session_ids",
    "split",
    "bucket",
    "split_key",
    "group_size",
)

#: Every key the tracked manifest may carry. ``sessions`` is deliberately
#: absent: per-session rows belong in the index, never here.
MANIFEST_FIELDS: tuple[str, ...] = (
    "manifest_version",
    "generator",
    "generated_at",
    "roots",
    "sources",
    "split_policy",
    "counts",
    "split_sha256",
    "holdout_sha256",
    "index",
    "index_sha256",
)

#: Longest a manifest or index string may be. Paths and ids are far shorter; a
#: value past this bound means transcript text leaked in.
MAX_FIELD_CHARS = 512

#: Head/tail records read per transcript while probing metadata.
PROBE_HEAD = 24
PROBE_TAIL = 8
#: A single record longer than this is kept as ``b""`` -- Claude's
#: ``file-history-snapshot`` records embed whole files and are never metadata.
MAX_PROBE_LINE = 1 << 20
#: Bounded window kept for the trailing records; the tail of a transcript is
#: small in every format measured.
TAIL_BYTES = 1 << 20
_CHUNK = 1 << 20


def bucket_for(split_key: str) -> int:
    """The documented bucket function: ``sha256(key) % 10``."""

    return int(hashlib.sha256(split_key.encode("utf-8")).hexdigest(), 16) % BUCKETS


def split_for_bucket(bucket: int) -> str:
    if bucket in TRAIN_BUCKETS:
        return "train"
    if bucket in EVAL_BUCKETS:
        return "eval"
    return "holdout"


def split_for(split_key: str) -> str:
    return split_for_bucket(bucket_for(split_key))


def split_digest(session_keys: Iterable[str]) -> str:
    """Seal digest over a split: sha256 of the sorted keys, one per line."""

    body = "".join(f"{key}\n" for key in sorted(set(session_keys)))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class _Union:
    """Union-find over identity tokens, rooted at the smallest token."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        parent = self._parent.setdefault(item, item)
        while parent != item:
            grand = self._parent.setdefault(parent, parent)
            self._parent[item] = grand  # path halving
            item, parent = parent, grand
        return item

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        low, high = sorted((root_left, root_right))
        self._parent[high] = low


@dataclass(slots=True)
class SessionEntry:
    """One row of the index."""

    session_key: str
    source: str
    cli: str
    path: str
    bytes: int = 0
    sha256: str = ""
    records: int = 0
    started_at: str | None = None
    ended_at: str | None = None
    cwd: str | None = None
    repo: str | None = None
    git_branch: str | None = None
    model: str | None = None
    cli_session_id: str | None = None
    linked_session_ids: tuple[str, ...] = ()
    split: str = ""
    bucket: int = -1
    split_key: str = ""
    group_size: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_key": self.session_key,
            "source": self.source,
            "cli": self.cli,
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "records": self.records,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "cwd": self.cwd,
            "repo": self.repo,
            "git_branch": self.git_branch,
            "model": self.model,
            "cli_session_id": self.cli_session_id,
            "linked_session_ids": list(self.linked_session_ids),
            "split": self.split,
            "bucket": self.bucket,
            "split_key": self.split_key,
            "group_size": self.group_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionEntry":
        return cls(
            session_key=str(data["session_key"]),
            source=str(data.get("source", "")),
            cli=str(data.get("cli", "")),
            path=str(data.get("path", "")),
            bytes=int(data.get("bytes", 0)),
            sha256=str(data.get("sha256", "")),
            records=int(data.get("records", 0)),
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            cwd=data.get("cwd"),
            repo=data.get("repo"),
            git_branch=data.get("git_branch"),
            model=data.get("model"),
            cli_session_id=data.get("cli_session_id"),
            linked_session_ids=tuple(data.get("linked_session_ids") or ()),
            split=str(data.get("split", "")),
            bucket=int(data.get("bucket", -1)),
            split_key=str(data.get("split_key", "")),
            group_size=int(data.get("group_size", 1)),
        )


@dataclass(frozen=True, slots=True)
class IndexStats:
    """What the manifest records about the index it sealed."""

    sha256: str
    lines: int
    bytes: int


@dataclass(slots=True)
class Corpus:
    """A built corpus: the tracked manifest plus the rows it seals."""

    manifest: dict[str, Any]
    entries: list[SessionEntry] = field(default_factory=list)


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------


class _HeadLines:
    """Collect the first ``limit`` complete records in bounded memory.

    A record longer than :data:`MAX_PROBE_LINE` is kept as ``b""`` rather than
    buffered: those are Claude's whole-file snapshots, never metadata, and the
    records after them still matter.
    """

    __slots__ = ("lines", "_buffer", "_overflow", "done", "_limit")

    def __init__(self, limit: int) -> None:
        self.lines: list[bytes] = []
        self._buffer = bytearray()
        self._overflow = False
        self.done = False
        self._limit = limit

    def feed(self, chunk: bytes) -> None:
        if self.done:
            return
        start = 0
        while len(self.lines) < self._limit:
            at = chunk.find(b"\n", start)
            if at == -1:
                rest = chunk[start:]
                if not self._overflow and len(self._buffer) + len(rest) <= MAX_PROBE_LINE:
                    self._buffer += rest
                else:
                    self._overflow = True
                    self._buffer.clear()
                return
            piece = chunk[start:at]
            if self._overflow or len(self._buffer) + len(piece) > MAX_PROBE_LINE:
                self.lines.append(b"")
            else:
                self.lines.append(bytes(self._buffer + piece))
            self._buffer.clear()
            self._overflow = False
            start = at + 1
        self.done = True

    def finish(self) -> list[bytes]:
        """Flush a trailing record that no newline ever closed."""

        if not self.done and (self._buffer or self._overflow):
            self.lines.append(b"" if self._overflow else bytes(self._buffer))
        return self.lines


def hash_and_probe(path: Path) -> tuple[str, int, int, list[bytes], list[bytes]]:
    """One streaming pass: sha256, byte count, record count, head and tail.

    Hashing already has to read every byte, so the head and tail records used
    for metadata come out of the same pass. Memory is bounded by
    :data:`MAX_PROBE_LINE` and :data:`TAIL_BYTES` regardless of file size --
    a 102 MB ``result.md`` is read in 1 MiB chunks and never materialized.
    """

    digest = hashlib.sha256()
    head = _HeadLines(PROBE_HEAD)
    tail = bytearray()
    total = 0
    newlines = 0
    last = b""
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            newlines += chunk.count(b"\n")
            last = chunk[-1:]
            head.feed(chunk)
            tail += chunk
            if len(tail) > TAIL_BYTES:
                del tail[: len(tail) - TAIL_BYTES]
    records = newlines + (1 if total and last != b"\n" else 0)
    head_lines = head.finish()[:PROBE_HEAD]
    tail_lines = bytes(tail).split(b"\n")
    if len(tail) < total:
        # the window opened mid-record
        tail_lines = tail_lines[1:]
    tail_lines = [line for line in tail_lines if line][-PROBE_TAIL:]
    return digest.hexdigest(), total, records, head_lines, tail_lines


def hash_prefix(path: Path, length: int) -> str:
    """sha256 of the first ``length`` bytes of ``path``.

    Used to tell an *append* (a resumed session wrote more records; the sealed
    bytes are still there, unchanged) from a *rewrite* (the sealed bytes no
    longer exist, which is what a moved seal looks like).
    """

    digest = hashlib.sha256()
    remaining = length
    with open(path, "rb") as handle:
        while remaining > 0:
            chunk = handle.read(min(_CHUNK, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _decode_records(raw_lines: Sequence[bytes]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in raw_lines:
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            continue
        try:
            value = json.loads(text)
        except ValueError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _probe_claude(entry: SessionEntry, head: Sequence[bytes], tail: Sequence[bytes]) -> None:
    for record in _decode_records(head):
        entry.started_at = entry.started_at or record.get("timestamp")
        entry.cwd = entry.cwd or record.get("cwd")
        entry.git_branch = entry.git_branch or record.get("gitBranch")
        message = record.get("message")
        if isinstance(message, dict) and message.get("model") not in (None, "<synthetic>"):
            entry.model = entry.model or str(message["model"])
    for record in reversed(_decode_records(tail)):
        if record.get("timestamp"):
            entry.ended_at = str(record["timestamp"])
            break


def _probe_codex(entry: SessionEntry, head: Sequence[bytes], tail: Sequence[bytes]) -> None:
    for record in _decode_records(head):
        entry.started_at = entry.started_at or record.get("timestamp")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        if record.get("type") == "session_meta":
            entry.cwd = entry.cwd or payload.get("cwd")
            git = payload.get("git")
            if isinstance(git, dict):
                entry.git_branch = entry.git_branch or git.get("branch")
        if payload.get("model"):
            entry.model = entry.model or str(payload["model"])
    for record in reversed(_decode_records(tail)):
        if record.get("timestamp"):
            entry.ended_at = str(record["timestamp"])
            break


def _probe_gigacode(entry: SessionEntry, head: Sequence[bytes], tail: Sequence[bytes]) -> None:
    for record in _decode_records(head):
        entry.started_at = entry.started_at or record.get("timestamp")
        entry.cwd = entry.cwd or record.get("cwd")
        if record.get("model"):
            entry.model = entry.model or str(record["model"])
    for record in reversed(_decode_records(tail)):
        if record.get("timestamp"):
            entry.ended_at = str(record["timestamp"])
            break


def _epoch_ms(value: Any) -> str | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return (
        datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _probe_ae_chat(
    entry: SessionEntry, path: Path, head: Sequence[bytes], tail: Sequence[bytes]
) -> None:
    for record in _decode_records(head):
        entry.started_at = entry.started_at or _epoch_ms(record.get("ts"))
        if record.get("model"):
            entry.model = entry.model or str(record["model"])
    for record in reversed(_decode_records(tail)):
        moment = _epoch_ms(record.get("ts"))
        if moment:
            entry.ended_at = moment
            break
    meta_path = path.parent / "meta.json"
    try:
        if meta_path.is_file() and meta_path.stat().st_size <= (4 << 20):
            with open(meta_path, "r", encoding="utf-8", errors="replace") as handle:
                meta = json.load(handle)
            if isinstance(meta, dict):
                entry.model = entry.model or meta.get("model")
                entry.cwd = entry.cwd or meta.get("workspace")
    except (OSError, ValueError):
        pass


def _probe_ae_node_result(entry: SessionEntry, head: Sequence[bytes]) -> None:
    """The CLI banner lives in the head records the hashing pass already kept."""

    lines = [raw.decode("utf-8", errors="replace") for raw in head]
    header = ADAPTERS["ae_node_result"].parse_header(lines)  # type: ignore[attr-defined]
    entry.model = header.get("model")
    entry.cwd = header.get("workdir")


def _probe_deepseek(entry: SessionEntry, path: Path) -> None:
    try:
        with _JsonMemberScanner(path) as scanner:
            metadata = scanner.value("metadata")
    except OSError:
        return
    if not isinstance(metadata, dict):
        return
    entry.model = metadata.get("model")
    entry.cwd = metadata.get("workspace")
    entry.started_at = metadata.get("created_at")
    entry.ended_at = metadata.get("updated_at")


def probe(ref: TranscriptRef) -> SessionEntry:
    """Hash a transcript and read the cheap metadata around its edges."""

    path = Path(ref.path)
    digest, size, records, head, tail = hash_and_probe(path)
    entry = SessionEntry(
        session_key=ref.session_key,
        source=ref.source,
        cli=ref.cli,
        path=str(path),
        bytes=size,
        sha256=digest,
        records=records,
        cli_session_id=ref.cli_session_id,
        linked_session_ids=tuple(sorted(set(ref.linked_session_ids))),
    )
    if ref.source == "claude":
        _probe_claude(entry, head, tail)
    elif ref.source == "codex":
        _probe_codex(entry, head, tail)
    elif ref.source == "gigacode":
        _probe_gigacode(entry, head, tail)
    elif ref.source == "ae_chat":
        _probe_ae_chat(entry, path, head, tail)
    elif ref.source == "ae_node_result":
        _probe_ae_node_result(entry, head)
    elif ref.source == "deepseek":
        _probe_deepseek(entry, path)
    entry.repo = repo_root(entry.cwd)
    return entry


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------


def assign_splits(entries: Sequence[SessionEntry]) -> list[SessionEntry]:
    """Union transcripts of one conversation, then bucket each group.

    Mutates and returns ``entries`` with ``split_key``/``bucket``/``split``/
    ``group_size`` filled in. The group's key is the union-find root, i.e. the
    smallest token in the component over session keys *and* the declared
    ``id:<uuid>`` identifiers -- see the module docstring for why the raw
    identifiers are part of the key rather than only the present members.
    """

    union = _Union()
    for entry in entries:
        union.find(entry.session_key)
        for identifier in entry.linked_session_ids:
            union.union(entry.session_key, f"id:{identifier}")
    groups: dict[str, list[SessionEntry]] = {}
    for entry in entries:
        groups.setdefault(union.find(entry.session_key), []).append(entry)
    for split_key, members in groups.items():
        bucket = bucket_for(split_key)
        label = split_for_bucket(bucket)
        for member in members:
            member.split_key = split_key
            member.bucket = bucket
            member.split = label
            member.group_size = len(members)
    return list(entries)


# --------------------------------------------------------------------------
# the index
# --------------------------------------------------------------------------


def index_line(entry: SessionEntry) -> bytes:
    """One compact index row. ``None`` and empty lists are omitted."""

    payload: dict[str, Any] = {}
    for key, value in entry.to_dict().items():
        if value is None or (isinstance(value, list) and not value):
            continue
        payload[key] = value
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return body.encode("utf-8") + b"\n"


def index_stats(entries: Iterable[SessionEntry]) -> IndexStats:
    """Digest the index without writing it, so the manifest can carry the seal."""

    digest = hashlib.sha256()
    lines = 0
    size = 0
    for entry in entries:
        line = index_line(entry)
        digest.update(line)
        lines += 1
        size += len(line)
    return IndexStats(sha256=digest.hexdigest(), lines=lines, bytes=size)


def write_index(entries: Iterable[SessionEntry], path: Path) -> IndexStats:
    """Write the per-session index, one compact JSON object per line."""

    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    lines = 0
    size = 0
    with open(path, "wb") as handle:
        for entry in entries:
            line = index_line(entry)
            assert_entry_no_content(json.loads(line))
            handle.write(line)
            digest.update(line)
            lines += 1
            size += len(line)
    return IndexStats(sha256=digest.hexdigest(), lines=lines, bytes=size)


def iter_index(path: Path) -> Iterator[SessionEntry]:
    """Stream the index. Thousands of rows, so it is never held twice."""

    with open(path, "rb") as handle:
        for raw in handle:
            text = raw.decode("utf-8").strip()
            if not text:
                continue
            yield SessionEntry.from_dict(json.loads(text))


def read_index(path: Path) -> list[SessionEntry]:
    return list(iter_index(path))


def hash_index_file(path: Path) -> IndexStats:
    """Digest the index *as written*, so a hand-edited row is caught."""

    digest = hashlib.sha256()
    lines = 0
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            lines += chunk.count(b"\n")
    return IndexStats(sha256=digest.hexdigest(), lines=lines, bytes=size)


def index_path_for(manifest_path: Path) -> Path:
    """The index that belongs to a manifest: its sibling ``*-index.jsonl``."""

    return manifest_path.with_name(f"{manifest_path.stem}-index.jsonl")


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


SPLIT_POLICY_RULE = "bucket = int(sha256(split_key).hexdigest(), 16) % 10"
SPLIT_KEY_DOC = (
    "union-find root over session keys and declared id:<uuid> identifiers; "
    "equal to the session's own key when it has no duplicate recording"
)
SEAL_DOC = 'sha256("\\n".join(sorted(session_keys)) + "\\n") per split'


def build_corpus(
    roots: Roots,
    *,
    sources: Iterable[str] | None = None,
    limit: int | None = None,
    generated_at: str | None = None,
    index_display_path: str | None = None,
    progress: Callable[[int, TranscriptRef], None] | None = None,
) -> Corpus:
    """Enumerate, hash and split every transcript under ``roots``."""

    entries: list[SessionEntry] = []
    seen: set[str] = set()
    skipped: Counter[str] = Counter()
    for ref in discover_transcripts(roots, sources):
        if limit is not None and len(entries) >= limit:
            break
        if ref.session_key in seen:
            skipped["duplicate_session_key"] += 1
            continue
        try:
            entry = probe(ref)
        except OSError:
            skipped["unreadable"] += 1
            continue
        if entry.bytes == 0:
            skipped["empty"] += 1
            continue
        seen.add(ref.session_key)
        entries.append(entry)
        if progress is not None:
            progress(len(entries), ref)
    assign_splits(entries)
    entries.sort(key=lambda item: item.session_key)
    stats = index_stats(entries)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "generator": GENERATOR,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "roots": roots.to_dict(),
        "sources": sorted(sources) if sources else sorted(ADAPTERS),
        "split_policy": {
            "buckets": BUCKETS,
            "rule": SPLIT_POLICY_RULE,
            "train": list(TRAIN_BUCKETS),
            "eval": list(EVAL_BUCKETS),
            "holdout": list(HOLDOUT_BUCKETS),
            "split_key": SPLIT_KEY_DOC,
            "seal": SEAL_DOC,
        },
        "counts": corpus_counts(entries, skipped),
        "split_sha256": {
            name: split_digest(e.session_key for e in entries if e.split == name)
            for name in SPLITS
        },
        "index": {
            "path": index_display_path or str(DEFAULT_INDEX),
            "lines": stats.lines,
            "bytes": stats.bytes,
            "fields": list(ENTRY_FIELDS),
            "tracked": False,
            "note": "rebuildable from roots; holds absolute paths, so it is gitignored",
        },
        "index_sha256": stats.sha256,
    }
    manifest["holdout_sha256"] = manifest["split_sha256"]["holdout"]
    assert_no_content(manifest, entries)
    return Corpus(manifest=manifest, entries=entries)


def corpus_counts(
    entries: Sequence[SessionEntry], skipped: Counter[str] | None = None
) -> dict[str, Any]:
    """The aggregates the tracked manifest carries in place of the rows."""

    by_split: Counter[str] = Counter(entry.split for entry in entries)
    by_source: Counter[str] = Counter(entry.source for entry in entries)
    by_cli: Counter[str] = Counter(entry.cli for entry in entries)
    split_source: dict[str, dict[str, int]] = {name: {} for name in SPLITS}
    for entry in entries:
        bucket = split_source.setdefault(entry.split, {})
        bucket[entry.source] = bucket.get(entry.source, 0) + 1
    return {
        "sessions": len(entries),
        "bytes": sum(entry.bytes for entry in entries),
        "records": sum(entry.records for entry in entries),
        "by_split": {name: by_split.get(name, 0) for name in SPLITS},
        "by_source": dict(sorted(by_source.items())),
        "by_cli": dict(sorted(by_cli.items())),
        "by_split_source": {
            name: dict(sorted(split_source.get(name, {}).items())) for name in SPLITS
        },
        "identity_groups": len({entry.split_key for entry in entries}),
        "sessions_in_multi_transcript_groups": sum(
            1 for entry in entries if entry.group_size > 1
        ),
        "largest_group": max((entry.group_size for entry in entries), default=0),
        "skipped": dict(sorted((skipped or Counter()).items())),
    }


class CorpusContentLeak(AssertionError):
    """A tracked artifact carried something other than metadata."""


def _check_strings(where: str, value: Any, key: str = "") -> None:
    if isinstance(value, str):
        if len(value) > MAX_FIELD_CHARS:
            raise CorpusContentLeak(
                f"{where} field {key!r} is {len(value)} chars "
                f"(limit {MAX_FIELD_CHARS})"
            )
        if "\n" in value or "\r" in value:
            raise CorpusContentLeak(f"{where} field {key!r} contains a newline")
    elif isinstance(value, dict):
        for name, item in value.items():
            _check_strings(where, name, key)
            _check_strings(where, item, f"{key}.{name}" if key else str(name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_strings(where, item, key)


def assert_entry_no_content(entry: dict[str, Any]) -> None:
    """Fail unless one index row is metadata only."""

    unknown = set(entry) - set(ENTRY_FIELDS)
    if unknown:
        raise CorpusContentLeak(
            f"index row {entry.get('session_key')!r} has non-schema fields: "
            f"{sorted(unknown)}"
        )
    for name, value in entry.items():
        _check_strings(f"index row {entry.get('session_key')!r}", value, name)


def assert_no_content(
    manifest: dict[str, Any], entries: Iterable[SessionEntry | dict[str, Any]] = ()
) -> None:
    """Fail unless the manifest, and every row given, are metadata only.

    The hard invariant of this stage is that no transcript content reaches a
    tracked artifact. Rather than trusting the writer, the whitelist is
    enforced here and re-checked by ``--verify``: the manifest against
    :data:`MANIFEST_FIELDS` (which has no room for per-session rows at all)
    and each index row against :data:`ENTRY_FIELDS`.
    """

    unknown = set(manifest) - set(MANIFEST_FIELDS)
    if unknown:
        raise CorpusContentLeak(f"manifest has non-schema fields: {sorted(unknown)}")
    for name, value in manifest.items():
        _check_strings("manifest", value, name)
    for entry in entries:
        assert_entry_no_content(
            entry.to_dict() if isinstance(entry, SessionEntry) else entry
        )


def write_manifest(manifest: dict[str, Any], path: Path) -> Path:
    """Persist the manifest deterministically (sorted keys, trailing newline)."""

    assert_no_content(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(payload + "\n", encoding="utf-8")
    return path


def write_corpus(
    corpus: Corpus, manifest_path: Path, index_path: Path | None = None
) -> tuple[Path, Path]:
    """Write the index first, then the manifest that seals it.

    The index is written before the manifest so a crash leaves the tracked
    artifact describing an index that does not exist rather than one that does
    not match -- ``--verify`` reports the first loudly and cannot be fooled by
    the second.
    """

    target_index = index_path or index_path_for(manifest_path)
    stats = write_index(corpus.entries, target_index)
    if stats.sha256 != corpus.manifest["index_sha256"]:
        raise CorpusContentLeak(
            "index digest changed between build and write: "
            f"{corpus.manifest['index_sha256']} != {stats.sha256}"
        )
    write_manifest(corpus.manifest, manifest_path)
    return manifest_path, target_index


def read_manifest(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def roots_from_manifest(manifest: dict[str, Any]) -> Roots:
    """Rebuild the scanned roots so ``--verify`` can re-derive without them."""

    data = manifest.get("roots") or {}

    def one(key: str) -> Path | None:
        value = data.get(key)
        return Path(value) if value else None

    return Roots(
        claude_projects=one("claude_projects"),
        codex_sessions=tuple(Path(p) for p in data.get("codex_sessions") or ()),
        gigacode_projects=one("gigacode_projects"),
        deepseek_sessions=one("deepseek_sessions"),
        ae_projects=one("ae_projects"),
    )


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def _blank_report() -> dict[str, Any]:
    return {
        "mode": "",
        "content_ok": True,
        "checked": 0,
        "verified": 0,
        "unchanged_bytes": 0,
        "changed": [],
        "missing": [],
        "split_mismatches": [],
        "holdout_appended": [],
        "holdout_rewritten": [],
        "holdout_missing": [],
        "index": {},
        "cross_check": {},
        "ok": True,
    }


def _cross_check(manifest: dict[str, Any], entries: Sequence[SessionEntry]) -> dict[str, Any]:
    """Re-derive the tracked digests and counts from the index rows."""

    recorded_counts = manifest.get("counts") or {}
    # ``skipped`` counts what never became a row, so it cannot be re-derived
    # from the index; carry it through so the rest of the comparison is real.
    recomputed_counts = corpus_counts(entries, Counter(recorded_counts.get("skipped") or {}))
    counts_diff = {
        key: {"recorded": recorded_counts.get(key), "recomputed": value}
        for key, value in recomputed_counts.items()
        if recorded_counts.get(key) != value
    }
    seal: dict[str, Any] = {}
    for name in SPLITS:
        recomputed = split_digest(e.session_key for e in entries if e.split == name)
        recorded = (manifest.get("split_sha256") or {}).get(name)
        seal[name] = {
            "recorded": recorded,
            "recomputed": recomputed,
            "match": recorded == recomputed,
        }
    holdout_alias = manifest.get("holdout_sha256") == seal["holdout"]["recomputed"]
    return {
        "counts_match": not counts_diff,
        "counts_differences": counts_diff,
        "seal": seal,
        "holdout_alias_match": holdout_alias,
        "ok": (not counts_diff)
        and all(seal[name]["match"] for name in SPLITS)
        and holdout_alias,
    }


def verify_corpus(
    manifest: dict[str, Any],
    *,
    index_path: Path | None = None,
    roots: Roots | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Check the seal, and report every disagreement with the disk.

    With the index present this does three different things, and they fail
    differently.

    *Cross-check* -- the counts, the per-split seal digests and every row's
    bucket are re-derived from the index and compared against the tracked
    manifest, and the index file's own sha256 is compared against
    ``index_sha256``. All of that depends on nothing but the recorded keys, so
    any mismatch is a corrupted or hand-edited artifact and always fails. This
    runs whether or not the byte-level check finds drift.

    *Seal* -- the recorded bytes of every **holdout** transcript must still be
    on disk. A file that merely grew is an ``append``: its recorded prefix
    still hashes to the recorded digest, which is what a resumed session looks
    like, and it is reported rather than failed. A file whose recorded prefix
    no longer hashes the same is a ``rewrite``; a holdout rewrite or
    disappearance fails, because that is a moved seal.

    *Environmental* drift in train/eval is reported only -- the Claude window
    rolls (~30 days), so those files legitimately come and go. ``strict=True``
    fails on any drift at all, in any split.

    Without an index there are no rows to check against, so the digests are
    re-derived from disk by rescanning ``roots`` and the differences are
    reported. That cannot distinguish a moved seal from a rolled window, so it
    reports rather than fails unless ``strict``.
    """

    report = _blank_report()
    try:
        assert_no_content(manifest)
    except CorpusContentLeak as error:
        report["content_ok"] = False
        report["content_error"] = str(error)
        report["ok"] = False

    if index_path is not None and Path(index_path).is_file():
        _verify_with_index(manifest, Path(index_path), report, strict=strict)
    else:
        _verify_by_rederiving(manifest, report, roots=roots, strict=strict)
    return report


def _verify_with_index(
    manifest: dict[str, Any], index_path: Path, report: dict[str, Any], *, strict: bool
) -> None:
    report["mode"] = "index"
    stats = hash_index_file(index_path)
    recorded = manifest.get("index_sha256")
    recorded_meta = manifest.get("index") or {}
    report["index"] = {
        "path": str(index_path),
        "present": True,
        "recorded_sha256": recorded,
        "actual_sha256": stats.sha256,
        "sha256_match": recorded == stats.sha256,
        "recorded_lines": recorded_meta.get("lines"),
        "actual_lines": stats.lines,
    }
    if not report["index"]["sha256_match"]:
        report["ok"] = False

    entries = read_index(index_path)
    try:
        assert_no_content(manifest, entries)
    except CorpusContentLeak as error:
        report["content_ok"] = False
        report["content_error"] = str(error)
        report["ok"] = False

    cross = _cross_check(manifest, entries)
    report["cross_check"] = cross
    if not cross["ok"]:
        report["ok"] = False

    report["checked"] = len(entries)
    for entry in entries:
        expected_bucket = bucket_for(entry.split_key or entry.session_key)
        expected_split = split_for_bucket(expected_bucket)
        if expected_bucket != entry.bucket or expected_split != entry.split:
            report["split_mismatches"].append(
                {
                    "session_key": entry.session_key,
                    "recorded": [entry.bucket, entry.split],
                    "recomputed": [expected_bucket, expected_split],
                }
            )
        path = Path(entry.path)
        if not path.is_file():
            report["missing"].append(entry.session_key)
            if entry.split == "holdout":
                report["holdout_missing"].append(entry.session_key)
            continue
        try:
            digest, size, _records, _head, _tail = hash_and_probe(path)
        except OSError:
            report["missing"].append(entry.session_key)
            if entry.split == "holdout":
                report["holdout_missing"].append(entry.session_key)
            continue
        if digest == entry.sha256 and size == entry.bytes:
            report["verified"] += 1
            report["unchanged_bytes"] += size
            continue
        appended = size > entry.bytes and hash_prefix(path, entry.bytes) == entry.sha256
        report["changed"].append(
            {
                "session_key": entry.session_key,
                "split": entry.split,
                "kind": "append" if appended else "rewrite",
                "recorded_sha256": entry.sha256,
                "actual_sha256": digest,
                "recorded_bytes": entry.bytes,
                "actual_bytes": size,
            }
        )
        if entry.split == "holdout":
            key = "holdout_appended" if appended else "holdout_rewritten"
            report[key].append(entry.session_key)

    if report["split_mismatches"]:
        report["ok"] = False
    if report["holdout_rewritten"] or report["holdout_missing"]:
        report["ok"] = False
    if strict and (report["changed"] or report["missing"]):
        report["ok"] = False


def _verify_by_rederiving(
    manifest: dict[str, Any], report: dict[str, Any], *, roots: Roots | None, strict: bool
) -> None:
    report["mode"] = "rederive"
    report["index"] = {"path": None, "present": False}
    # Nothing binds the manifest to rows any more, so the one check that still
    # stands on its own is its internal consistency.
    alias_ok = manifest.get("holdout_sha256") == (manifest.get("split_sha256") or {}).get(
        "holdout"
    )
    report["cross_check"] = {"holdout_alias_match": alias_ok, "ok": alias_ok}
    if not alias_ok:
        report["ok"] = False

    scan_roots = roots or roots_from_manifest(manifest)
    rebuilt = build_corpus(
        scan_roots,
        sources=manifest.get("sources") or None,
        generated_at=manifest.get("generated_at"),
        index_display_path=(manifest.get("index") or {}).get("path"),
    )
    recorded_counts = manifest.get("counts") or {}
    fresh_counts = rebuilt.manifest["counts"]
    drift = {
        key: {"recorded": recorded_counts.get(key), "rederived": fresh_counts.get(key)}
        for key in ("sessions", "bytes", "records", "by_split", "by_source")
        if recorded_counts.get(key) != fresh_counts.get(key)
    }
    seal = {
        name: {
            "recorded": (manifest.get("split_sha256") or {}).get(name),
            "rederived": rebuilt.manifest["split_sha256"][name],
            "match": (manifest.get("split_sha256") or {}).get(name)
            == rebuilt.manifest["split_sha256"][name],
        }
        for name in SPLITS
    }
    report["checked"] = fresh_counts["sessions"]
    report["verified"] = fresh_counts["sessions"]
    report["rederived"] = {
        "counts": fresh_counts,
        "counts_drift": drift,
        "seal": seal,
        "index_sha256": rebuilt.manifest["index_sha256"],
        "index_sha256_match": rebuilt.manifest["index_sha256"] == manifest.get("index_sha256"),
        "note": (
            "no index on disk: the rolling Claude window makes drift expected, "
            "so it is reported; rebuild with --build to re-seal"
        ),
    }
    if strict and (drift or not all(seal[name]["match"] for name in SPLITS)):
        report["ok"] = False


# --------------------------------------------------------------------------
# consumption helpers for downstream children
# --------------------------------------------------------------------------


def refs_from_index(
    index_path: Path, *, splits: Iterable[str] | None = None
) -> list[TranscriptRef]:
    """Rebuild loadable refs for the requested splits.

    Downstream children call this with ``splits=("train",)`` or
    ``("eval",)``. Passing ``("holdout",)`` is possible on purpose: the field
    measurement child needs it exactly once, and the seal records who moved it.
    """

    wanted = set(splits) if splits else set(SPLITS)
    return [ref_for(entry) for entry in iter_index(Path(index_path)) if entry.split in wanted]


def ref_for(entry: SessionEntry) -> TranscriptRef:
    return TranscriptRef(
        source=entry.source,
        cli=entry.cli,
        path=Path(entry.path),
        session_key=entry.session_key,
        cli_session_id=entry.cli_session_id,
        linked_session_ids=entry.linked_session_ids,
    )


def load_session(entry: SessionEntry) -> SessionRecord:
    """Parse the transcript behind one index row, pinned to its hash."""

    return load_transcript(ref_for(entry), sha256=entry.sha256)


def recovery_stats(records: Iterable[SessionRecord]) -> dict[str, Any]:
    """Aggregate what the normalizer actually recovered from a set of sessions."""

    totals: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = {}
    positions: dict[str, list[float]] = {"recall": [], "write": []}
    for record in records:
        bucket = by_source.setdefault(record.source, Counter())
        for name, value in (
            ("sessions", 1),
            ("turns", len(record.turns)),
            ("tool_calls", len(record.tool_calls)),
            ("file_mutations", len(record.file_mutations)),
            ("recalls", len(record.recalls)),
            ("recall_event_ids", len(record.recall_event_ids)),
            ("delivered_nodes", len(record.delivered_node_ids)),
            ("writes", len(record.writes)),
            ("written_nodes", len(record.written_node_ids)),
            ("sessions_with_recall", 1 if record.recalls else 0),
            ("sessions_with_write", 1 if record.writes else 0),
            ("sessions_with_diff", 1 if record.file_mutations else 0),
        ):
            totals[name] += value
            bucket[name] += value
        positions["recall"].extend(item.position for item in record.recalls)
        positions["write"].extend(item.position for item in record.writes)
    return {
        "totals": dict(sorted(totals.items())),
        "by_source": {name: dict(sorted(c.items())) for name, c in sorted(by_source.items())},
        "median_position": {
            name: (_median(values) if values else None) for name, values in positions.items()
        },
    }


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[middle], 4)
    return round((ordered[middle - 1] + ordered[middle]) / 2, 4)
