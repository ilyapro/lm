"""Offline **end-to-end** retrieval harness over a frozen database snapshot.

``living_memory.replay`` re-ranks the *recorded* per-result method scores of
past ``recall_events`` (``replay.build_candidates`` / ``replay.rerank_event``).
That makes it the right arbiter for weight-mixing policy, and structurally the
wrong one for any change that alters the scores themselves or the candidate
set — chunked embeddings change both. This module therefore runs the *real*
retrieval path (``MemoryRecallService.memory_recall``) against a snapshot and
scores the result with replay's labeling and metric machinery, which it
imports rather than forks (``_containment``, ``build_label_data``,
``LabelConfig``, ``MetricAccumulator``, ``_ratio``).

Snapshot discipline, two levels
-------------------------------

``MemoryStore.__init__`` opens ``sqlite3.connect(str(path))`` with no
``mode=ro`` and immediately runs ``_initialize_schema()`` plus
``_seed_retrieval_weights()``, so a live ``MemoryRecallService`` *cannot* sit
on a read-only connection. Hence:

1. ``snapshot`` opens ``--source-db`` as ``file:...?mode=ro`` (``uri=True``)
   and copies it with :meth:`sqlite3.Connection.backup` into ``--snapshot-out``,
   plus a sidecar manifest (sha256, row counts, event span, source identity).
   A plain ``cp`` is wrong: the live DB carries a large ``-wal``, and a file
   copy without it is incomplete; the backup API yields the logical state
   including WAL. The source is never opened through ``MemoryStore``.
2. ``run`` and ``build-goldset`` take that frozen snapshot and make their own
   fresh temporary working copy from it (same backup API) before constructing
   ``MemoryStore``, then delete the copy. Repeated runs therefore always start
   from identical bytes and the frozen snapshot stays pristine.

Both entry points also accept ``--source-db`` pointing straight at the live
database; they then freeze a temporary snapshot first and proceed identically.

Tail rule (mechanical, auditable)
---------------------------------

.. code-block:: text

%(tail_rule)s

Where tail items actually come from (measured 2026-08-17 on the live DB)
-----------------------------------------------------------------------

``content_grounded`` items are effectively **never** tail, and this is
structural rather than accidental: 0 of 120 sampled items were tail. The
containment label selects results that share a lot of vocabulary with the
consuming trace (7-95 shared tokens per node in that sample), and with that
many grounding tokens spread through the node at least one always lands
inside the visible prefix, whose earliest grounding offset came out at
0-47 characters against visible spans of 167-472. Do not expect the tail
subset to fill itself from recorded events.

The tail subset therefore has to be **curated**, through ``--seed-queries``,
where the justifying text is the query itself and the author controls which
tokens it shares with the node. There is ample room for that: 430 of the 433
active ``level:schema`` nodes carry more than 200 characters past their
visible span (median content 2074 chars, median visible share 0.19), and
picking query terms that occur only past ``visible_char_end`` yields
``tail=True`` mechanically — verified on three such nodes.

Query anchors: three buckets and one ablation switch
----------------------------------------------------

``run`` annotates every goldset item against the snapshot's anchor corpus
(:func:`annotate_anchors`) and turns that into first-class metric buckets
beside ``per_stratum`` and ``tail``:

``anchor_holdout``
    ``exact_repeat`` versus ``fingerprint_disjoint``. Reported apart, never
    merged: an exact repeat pulling back its own past answer is a lookup
    table, and only the disjoint subset can carry a generalization claim.

``fresh_node_visibility``
    The self-reinforcement guard. Anchors, grounding and usefulness form a
    loop; this bucket is what says the loop does not bury new memory. It
    scores *only* the relevant nodes created after the matched anchor last
    learned anything, and must not be worse with anchors than without.

``anchor_coverage``
    How much of the goldset the corpus can even see, including the unfloored
    nearest-anchor cosine, which separates "no anchor exists for this class of
    query" from "one exists and the match floor rejected it".

``--no-anchors`` flips ``MemoryRecallService(anchor_seeding=False)`` and
nothing else, so the two arms of a leak-free with/without run hold one
snapshot, one goldset and one code path fixed. The annotation deliberately
ignores the switch: both arms must be scored over the same bucket membership.

Determinism
-----------

The goldset is iterated sorted by ``query_id``, samples are drawn with
``random.Random(--seed)`` over a sorted id list, the event slice is pinned by
an explicit ``--cutoff``, every dumped mapping is key-sorted, and wall-clock
lives in ``provenance`` only. Two invocations over the same frozen snapshot
and goldset produce byte-identical ``metrics`` blocks.

CLI::

    python3 -m living_memory.retrieval_harness snapshot --source-db DB --snapshot-out SNAP
    python3 -m living_memory.retrieval_harness build-goldset --snapshot SNAP --cutoff ISO \\
        --seed 0 --seed-queries seed-queries.json --out goldset.jsonl
    python3 -m living_memory.retrieval_harness [run] --snapshot SNAP --goldset goldset.jsonl \\
        --report baseline.json --markdown baseline.md [--no-anchors]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from living_memory.config import MemoryConfig
from living_memory.embeddings import LocalEmbeddingModel, cosine_similarity, tokenize
from living_memory.models import NODE_LEVELS
from living_memory.query_anchors import (
    ANCHOR_MATCH_COSINE_THRESHOLD,
    ANCHOR_MATCH_LIMIT,
    match_anchors,
)
from living_memory.replay import (
    HIT_KS,
    LabelConfig,
    MetricAccumulator,
    ReplayEvent,
    ReplayResult,
    _containment,
    _ratio,
    build_label_data,
    load_replay_events,
    normalize_cutoff,
    open_readonly,
)
from living_memory.retrieval import MemoryRecallService, RecallResult
from living_memory.scope import normalize_scope
from living_memory.storage import MemoryStore, recall_fingerprint

#: v2 adds the anchor annotation and the three buckets it feeds
#: (``fresh_node_visibility``, ``anchor_holdout``, ``anchor_coverage``), plus
#: per-stratum channel attribution. Every v1 key keeps its meaning.
HARNESS_VERSION = 2

STRATA: tuple[str, ...] = ("content_grounded", "cross_lingual", "role_query")
CHANNELS: tuple[str, ...] = ("bm25", "vector", "graph", "trigger")

GOLDSET_FIELDS: tuple[str, ...] = (
    "query_id",
    "query",
    "scope",
    "ambient_context",
    "depth",
    "max_results",
    "relevant_node_ids",
    "stratum",
    "tail",
    "source_event_id",
    "provenance",
)

#: ``max_seq_length`` of ``paraphrase-multilingual-MiniLM-L12-v2`` as configured
#: by sentence-transformers. Read-only here: the harness never changes it.
MAX_SEQ_LENGTH = 128

#: Minimum ``max_results`` a content-grounded item is replayed with, so that
#: ``hit@10`` is well defined for every goldset item.
MIN_MAX_RESULTS = 10

DEFAULT_AGREEMENT_SAMPLE = 20
AGREEMENT_TOP_K = 5

MANIFEST_SUFFIX = ".manifest.json"

TAIL_RULE = """\
Per relevant node of a goldset item:

1. grounding tokens := tokenize(node.content) & tokenize(justifying_text),
   where `tokenize` is `living_memory.embeddings.tokenize` and the justifying
   text is the consuming trace's content for the `content_grounded` stratum
   and the goldset query itself for `cross_lingual` and `role_query`.
2. visible span := encode node.content with the real model tokenizer using
   truncation=True, max_length=128, return_offsets_mapping=True; drop every
   token whose get_special_tokens_mask() entry is 1; visible_char_end is the
   maximum end offset of what remains. That is the character prefix the
   embedding vector actually sees.
3. The node is TAIL iff its grounding-token set is non-empty AND every
   occurrence of every grounding token in node.content starts at character
   offset >= visible_char_end. An empty grounding-token set (the
   cross_lingual case by construction) is NOT tail, and the reason is
   recorded in the item's provenance.

Item level: an item is TAIL iff at least one relevant node has a non-empty
grounding-token set and every such node is TAIL. A single-relevant-node item
therefore reduces to the per-node rule exactly; an item that keeps any
head-grounded relevant node is not tail, because that node is still reachable
through the prefix the vector sees."""

PER_CHANNEL_RULE = """\
A per-channel bucket re-orders the SAME returned result list by that single
channel score descending, ties broken by the live rank, and computes
hit@1/hit@5/hit@10/MRR over that order. No result is added or removed, so the
bucket measures how well one channel alone would have ranked what retrieval
actually returned. `channel_attribution` is a separate view: for the first
relevant result in the LIVE order it counts each method present in that
result's `.methods`, so a result found by two channels counts once for each."""

AGREEMENT_RULE = """\
For a `random.Random(seed)` sample of goldset query ids (drawn over the sorted
id list), a second `MemoryStore` + `MemoryRecallService` pair is constructed
independently of the runner's code path on the same working copy, and its
top-5 node ids are compared with the runner's top-5. `agreement_rate` is the
share of sampled items that match exactly; every mismatch is listed in full.
The run exits non-zero when the rate is below 1.0 unless `--divergence-note`
explains it."""

ANCHOR_SLICE_RULE = """\
Per goldset item, computed from the snapshot alone so both arms of an
anchors-on/anchors-off comparison annotate identically:

1. The query is embedded with the run's own encoder and matched against the
   live anchor vectors of the item's resolved `ScopePlan` through
   `query_anchors.match_anchors`, at the shipped floor
   (ANCHOR_MATCH_COSINE_THRESHOLD, limit ANCHOR_MATCH_LIMIT) -- the same call,
   scope gate and floor `retrieval._collect_anchor_seeds` uses. The annotation
   therefore states what retrieval could see, not what a laxer probe finds.
2. An item is EXACT_REPEAT iff `storage.recall_fingerprint(query, scope)`
   already names a live anchor in one of those scopes, and
   FINGERPRINT_DISJOINT otherwise. Both subsets are reported apart, never
   merged into a headline: an exact repeat retrieving its own past answer is a
   lookup table, and the generalization claim lives only in the disjoint
   subset, where a new query must reach the right node by resembling a
   *different* past query.
3. `anchor_as_of` := the newest `updated_at` over the matched anchors -- the
   moment the anchor corpus last learned anything about this query. The
   backfill stamps it with the source event's own `created_at`, so it is a
   training-set timestamp and not a run timestamp.
4. A relevant node is FRESH iff its `created_at` is strictly after
   `anchor_as_of` and it is not itself one of the matched anchors' live edge
   targets. An item enters the FRESH-NODE-VISIBILITY slice iff it matched an
   anchor, that anchor holds at least one live edge (it points at an older
   node), and the item has at least one fresh relevant node. The slice scores
   ONLY those fresh nodes: it asks whether the node the anchor cannot know
   about is still reachable once the anchor's older targets are seeded into
   the graph channel. This is the gate against "the rich get richer"; anchors
   must not make it worse than the anchor-free arm."""

if __doc__:  # `python -OO` strips docstrings; the rule still lives in TAIL_RULE.
    __doc__ = __doc__ % {"tail_rule": TAIL_RULE}


# ---------------------------------------------------------------------------
# Snapshot machinery
# ---------------------------------------------------------------------------

_SNAPSHOT_COUNTS: tuple[tuple[str, str], ...] = (
    ("nodes", "SELECT COUNT(*) FROM nodes"),
    # `nodes` has no superseded_by column: active means decayed = 0.
    ("active_nodes", "SELECT COUNT(*) FROM nodes WHERE decayed = 0"),
    ("connections", "SELECT COUNT(*) FROM connections"),
    ("recall_events", "SELECT COUNT(*) FROM recall_events"),
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def backup_database(source: str | Path, target: str | Path) -> None:
    """Copy ``source`` to ``target`` through the SQLite backup API.

    The source is opened strictly read-only (``file:...?mode=ro``, ``uri=True``)
    and never through ``MemoryStore``. The backup API transfers the *logical*
    database, so the WAL contents come along and no separate ``-wal`` file is
    needed at the destination — unlike a plain file copy, which is incomplete
    whenever the source has an active WAL.
    """

    source_path = Path(source)
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    reader = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    try:
        writer = sqlite3.connect(str(target_path))
        try:
            reader.backup(writer)
        finally:
            writer.close()
    finally:
        reader.close()


def manifest_path_for(snapshot: str | Path) -> Path:
    snapshot_path = Path(snapshot)
    return snapshot_path.with_name(snapshot_path.name + MANIFEST_SUFFIX)


def snapshot_row_counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {name: int(connection.execute(sql).fetchone()[0]) for name, sql in _SNAPSHOT_COUNTS}


def describe_snapshot(
    snapshot: str | Path,
    *,
    source: str | Path | None = None,
    captured_at: str | None = None,
) -> dict[str, Any]:
    """Manifest contents for an existing snapshot file (no side effects)."""

    snapshot_path = Path(snapshot)
    connection = open_readonly(snapshot_path)
    try:
        counts = snapshot_row_counts(connection)
        span = connection.execute(
            "SELECT MIN(created_at) AS lo, MAX(created_at) AS hi FROM recall_events"
        ).fetchone()
    finally:
        connection.close()
    source_path = Path(source) if source is not None else None
    return {
        "harness_version": HARNESS_VERSION,
        "captured_at": captured_at or utc_now_iso(),
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": sha256_file(snapshot_path),
        "snapshot_bytes": snapshot_path.stat().st_size,
        "source_path": str(source_path) if source_path is not None else None,
        "source_bytes": source_path.stat().st_size if source_path is not None else None,
        "row_counts": counts,
        "recall_events_created_at": {
            "min": span["lo"] if span is not None else None,
            "max": span["hi"] if span is not None else None,
        },
    }


def create_snapshot(
    source: str | Path,
    snapshot_out: str | Path,
    *,
    captured_at: str | None = None,
) -> dict[str, Any]:
    """Freeze ``source`` into ``snapshot_out`` and write the sidecar manifest."""

    backup_database(source, snapshot_out)
    manifest = describe_snapshot(snapshot_out, source=source, captured_at=captured_at)
    manifest_path_for(snapshot_out).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_manifest(snapshot: str | Path) -> dict[str, Any]:
    """Sidecar manifest if present, else one computed from the snapshot itself."""

    path = manifest_path_for(snapshot)
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return loaded
    return describe_snapshot(snapshot)


@contextlib.contextmanager
def working_copy(snapshot: str | Path) -> Iterator[Path]:
    """Fresh temp copy of the frozen snapshot, deleted on exit.

    ``MemoryStore`` migrates and writes whatever it opens, so every run gets
    its own copy: identical starting bytes for repeated runs, and the frozen
    snapshot itself is never touched.
    """

    directory = Path(tempfile.mkdtemp(prefix="lm-retrieval-harness-"))
    try:
        target = directory / "working.sqlite3"
        backup_database(snapshot, target)
        yield target
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@contextlib.contextmanager
def frozen_snapshot(
    snapshot: str | Path | None, source_db: str | Path | None
) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield ``(frozen snapshot path, manifest)``.

    With ``--snapshot`` the caller's frozen file is used as-is. With
    ``--source-db`` (the live database is a legal argument) a temporary
    snapshot is frozen first, so the run still starts from immutable bytes.
    """

    if snapshot is not None:
        snapshot_path = Path(snapshot)
        yield snapshot_path, load_manifest(snapshot_path)
        return
    if source_db is None:  # pragma: no cover - argparse enforces this
        raise ValueError("one of --snapshot / --source-db is required")
    directory = Path(tempfile.mkdtemp(prefix="lm-retrieval-snapshot-"))
    try:
        target = directory / "snapshot.sqlite3"
        manifest = create_snapshot(source_db, target)
        yield target, manifest
    finally:
        shutil.rmtree(directory, ignore_errors=True)


# ---------------------------------------------------------------------------
# Goldset: schema, validation, loading
# ---------------------------------------------------------------------------


class GoldsetError(ValueError):
    """Schema violation in a goldset or seed-query file (message names the line)."""


@dataclass(frozen=True, slots=True)
class GoldsetItem:
    """One evaluation query with its relevance judgement."""

    query_id: str
    query: str
    scope: str | None
    ambient_context: dict[str, Any] | None
    depth: int | str | None
    max_results: int
    relevant_node_ids: tuple[str, ...]
    stratum: str
    tail: bool
    source_event_id: str | None
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "scope": self.scope,
            "ambient_context": self.ambient_context,
            "depth": self.depth,
            "max_results": self.max_results,
            "relevant_node_ids": list(self.relevant_node_ids),
            "stratum": self.stratum,
            "tail": self.tail,
            "source_event_id": self.source_event_id,
            "provenance": self.provenance,
        }


def validate_goldset_record(raw: Any, line_number: int) -> GoldsetItem:
    """Validate one goldset record, naming the offending line on failure."""

    def fail(message: str) -> None:
        raise GoldsetError(f"goldset line {line_number}: {message}")

    if not isinstance(raw, dict):
        fail("expected a JSON object")
    keys = set(raw)
    missing = sorted(set(GOLDSET_FIELDS) - keys)
    if missing:
        fail(f"missing key(s): {', '.join(missing)}")
    unknown = sorted(keys - set(GOLDSET_FIELDS))
    if unknown:
        fail(f"unknown key(s): {', '.join(unknown)}")

    query_id = raw["query_id"]
    if not isinstance(query_id, str) or not query_id.strip():
        fail("query_id must be a non-empty string")
    query = raw["query"]
    if not isinstance(query, str) or not query.strip():
        fail("query must be a non-empty string")
    scope = raw["scope"]
    if scope is not None and not isinstance(scope, str):
        fail("scope must be a string or null")
    ambient_context = raw["ambient_context"]
    if ambient_context is not None and not isinstance(ambient_context, dict):
        fail("ambient_context must be an object or null")
    depth = raw["depth"]
    if isinstance(depth, bool) or not (depth is None or isinstance(depth, (int, str))):
        fail("depth must be an integer, a string, or null")
    max_results = raw["max_results"]
    if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results <= 0:
        fail("max_results must be a positive integer")
    relevant = raw["relevant_node_ids"]
    if not isinstance(relevant, list) or not relevant:
        fail("relevant_node_ids must be a non-empty list")
    if any(not isinstance(node_id, str) or not node_id.strip() for node_id in relevant):
        fail("relevant_node_ids must contain non-empty strings")
    if len(set(relevant)) != len(relevant):
        fail("relevant_node_ids must not repeat a node id")
    stratum = raw["stratum"]
    if stratum not in STRATA:
        fail(f"unknown stratum {stratum!r}; expected one of {', '.join(STRATA)}")
    tail = raw["tail"]
    if not isinstance(tail, bool):
        fail("tail must be a boolean")
    source_event_id = raw["source_event_id"]
    if source_event_id is not None and not isinstance(source_event_id, str):
        fail("source_event_id must be a string or null")
    provenance = raw["provenance"]
    if not isinstance(provenance, dict):
        fail("provenance must be an object")

    return GoldsetItem(
        query_id=query_id,
        query=query,
        scope=scope,
        ambient_context=dict(ambient_context) if ambient_context is not None else None,
        depth=depth,
        max_results=max_results,
        relevant_node_ids=tuple(relevant),
        stratum=stratum,
        tail=tail,
        source_event_id=source_event_id,
        provenance=dict(provenance),
    )


def load_goldset(path: str | Path) -> list[GoldsetItem]:
    """Parse and validate a goldset JSONL file; always sorted by ``query_id``."""

    items: list[GoldsetItem] = []
    seen: dict[str, int] = {}
    text = Path(path).read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except ValueError as exc:
            raise GoldsetError(f"goldset line {line_number}: invalid JSON ({exc})") from exc
        item = validate_goldset_record(raw, line_number)
        first_seen = seen.get(item.query_id)
        if first_seen is not None:
            raise GoldsetError(
                f"goldset line {line_number}: duplicate query_id {item.query_id!r} "
                f"(first seen on line {first_seen})"
            )
        seen[item.query_id] = line_number
        items.append(item)
    if not items:
        raise GoldsetError(f"goldset {path} contains no items")
    items.sort(key=lambda item: item.query_id)
    return items


def dump_goldset(items: Sequence[GoldsetItem], path: str | Path) -> None:
    """Write items as JSONL, sorted by ``query_id`` with key-sorted objects."""

    ordered = sorted(items, key=lambda item: item.query_id)
    lines = [
        json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) for item in ordered
    ]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Tail classification
# ---------------------------------------------------------------------------


class VisibleSpanTokenizer(Protocol):
    """Anything that can say how far into a text the encoder actually looks."""

    def visible_char_end(self, text: str) -> int: ...


class ModelVisibleSpan:
    """Visible span from the *real* tokenizer, loaded lazily on first use.

    Kept behind an injectable interface so tests can substitute a stub and
    never import torch / sentence-transformers.
    """

    def __init__(
        self, *, max_length: int = MAX_SEQ_LENGTH, model_name: str | None = None
    ) -> None:
        self._max_length = int(max_length)
        self._model_name = model_name
        self._tokenizer: Any | None = None

    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            kwargs = {"model_name": self._model_name} if self._model_name else {}
            model = LocalEmbeddingModel(**kwargs)
            model.warmup()
            backend = getattr(model, "_model", None)
            if backend is None:
                raise RuntimeError(
                    "the real sentence-transformers tokenizer is unavailable "
                    "(LIVING_MEMORY_EMBEDDING_BACKEND selects the hash backend, or the "
                    "model is not cached locally); inject a VisibleSpanTokenizer instead"
                )
            self._tokenizer = backend.tokenizer
        return self._tokenizer

    def visible_char_end(self, text: str) -> int:
        if not text:
            return 0
        tokenizer = self.tokenizer()
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=self._max_length,
            return_offsets_mapping=True,
        )
        specials = tokenizer.get_special_tokens_mask(
            encoded["input_ids"], already_has_special_tokens=True
        )
        ends = [
            int(end)
            for (_start, end), special in zip(encoded["offset_mapping"], specials, strict=False)
            if not special
        ]
        return max(ends, default=0)


_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_TOKEN_SEPARATORS = ("_", "-", "/", ".")


def _is_ascii_lower(char: str) -> bool:
    return "a" <= char <= "z"


def _is_ascii_upper(char: str) -> bool:
    return "A" <= char <= "Z"


def _camel_pieces(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split ``text[start:end]`` at every ASCII lower->upper boundary."""

    pieces: list[tuple[int, int]] = []
    piece_start = start
    for index in range(start + 1, end):
        if _is_ascii_lower(text[index - 1]) and _is_ascii_upper(text[index]):
            pieces.append((piece_start, index))
            piece_start = index
    pieces.append((piece_start, end))
    return pieces


def token_spans(text: str) -> list[tuple[str, int, int]]:
    """``tokenize(text)`` re-derived as ``(token, start, end)`` char spans.

    ``embeddings.tokenize`` splits camelCase by *inserting* a space, which
    shifts every later offset; here the same split is applied inside each word
    span instead, so ``(start, end)`` still index ``text``. Every candidate
    span is fed back through ``tokenize`` itself, so the emitted tokens are
    exactly the tokens ``tokenize(text)`` produces, in the same order.
    """

    separated = text
    for separator in _TOKEN_SEPARATORS:
        separated = separated.replace(separator, " ")
    spans: list[tuple[str, int, int]] = []
    for match in _WORD_RE.finditer(separated):
        start, end = match.span()
        for piece_start, piece_end in _camel_pieces(separated, start, end):
            for token in tokenize(text[piece_start:piece_end]):
                spans.append((token, piece_start, piece_end))
    return spans


def grounding_tokens(node_content: str, justifying_text: str) -> tuple[str, ...]:
    """Tokens shared by the node content and the text that justifies it."""

    shared = frozenset(tokenize(node_content)) & frozenset(tokenize(justifying_text))
    return tuple(sorted(shared))


TAIL_REASON_NO_GROUNDING = "no_grounding_tokens"
TAIL_REASON_BEYOND = "grounding_beyond_visible_span"
TAIL_REASON_WITHIN = "grounding_within_visible_span"

ITEM_TAIL_REASON_NO_GROUNDING = "no_grounded_relevant_node"
ITEM_TAIL_REASON_TAIL = "every_grounded_relevant_node_beyond_visible_span"
ITEM_TAIL_REASON_HEAD = "some_grounded_relevant_node_within_visible_span"


@dataclass(frozen=True, slots=True)
class TailVerdict:
    """Per-node outcome of the documented tail rule."""

    tail: bool
    reason: str
    visible_char_end: int
    earliest_grounding_char: int | None
    grounding_tokens: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tail": self.tail,
            "reason": self.reason,
            "visible_char_end": self.visible_char_end,
            "earliest_grounding_char": self.earliest_grounding_char,
            "grounding_tokens": list(self.grounding_tokens),
        }


def classify_node_tail(
    node_content: str, justifying_text: str, *, tokenizer: VisibleSpanTokenizer
) -> TailVerdict:
    """Apply the documented per-node tail rule (see ``TAIL_RULE``)."""

    tokens = grounding_tokens(node_content, justifying_text)
    visible_char_end = tokenizer.visible_char_end(node_content)
    if not tokens:
        return TailVerdict(False, TAIL_REASON_NO_GROUNDING, visible_char_end, None, ())
    wanted = set(tokens)
    starts = [start for token, start, _end in token_spans(node_content) if token in wanted]
    if not starts:
        # Defensive: grounding tokens come from the same text, so this only
        # happens if span re-derivation and tokenize ever drift apart.
        return TailVerdict(False, TAIL_REASON_NO_GROUNDING, visible_char_end, None, tokens)
    earliest = min(starts)
    is_tail = earliest >= visible_char_end
    return TailVerdict(
        is_tail,
        TAIL_REASON_BEYOND if is_tail else TAIL_REASON_WITHIN,
        visible_char_end,
        earliest,
        tokens,
    )


def classify_item_tail(
    node_contents: Mapping[str, str],
    node_ids: Sequence[str],
    justifying_text: str,
    *,
    tokenizer: VisibleSpanTokenizer,
) -> tuple[bool, dict[str, Any]]:
    """Item-level tail verdict plus the audit trail for its provenance."""

    verdicts = {
        node_id: classify_node_tail(
            node_contents.get(node_id, ""), justifying_text, tokenizer=tokenizer
        )
        for node_id in sorted(node_ids)
    }
    grounded = [verdict for verdict in verdicts.values() if verdict.grounding_tokens]
    is_tail = bool(grounded) and all(verdict.tail for verdict in grounded)
    if not grounded:
        reason = ITEM_TAIL_REASON_NO_GROUNDING
    elif is_tail:
        reason = ITEM_TAIL_REASON_TAIL
    else:
        reason = ITEM_TAIL_REASON_HEAD
    audit = {
        "rule": TAIL_RULE,
        "reason": reason,
        "max_seq_length": MAX_SEQ_LENGTH,
        "nodes": {node_id: verdict.to_dict() for node_id, verdict in verdicts.items()},
    }
    return is_tail, audit


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HarnessResult:
    """One live result with its per-channel evidence."""

    node_id: str
    rank: int
    level: str
    scope: str
    score: float
    bm25_score: float
    vector_score: float
    graph_score: float
    trigger_score: float
    methods: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "rank": self.rank,
            "level": self.level,
            "scope": self.scope,
            "score": self.score,
            "bm25_score": self.bm25_score,
            "vector_score": self.vector_score,
            "graph_score": self.graph_score,
            "trigger_score": self.trigger_score,
            "methods": list(self.methods),
        }


@dataclass(frozen=True, slots=True)
class HarnessRun:
    """One goldset item replayed end-to-end."""

    item: GoldsetItem
    results: tuple[HarnessResult, ...]

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(result.node_id for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.item.query_id,
            "stratum": self.item.stratum,
            "tail": self.item.tail,
            "relevant_node_ids": list(self.item.relevant_node_ids),
            "ranked_node_ids": list(self.node_ids),
            "results": [result.to_dict() for result in self.results],
        }


def _harness_result(index: int, result: RecallResult) -> HarnessResult:
    return HarnessResult(
        node_id=result.node_id,
        rank=index + 1,
        level=str(result.node.level),
        scope=str(result.node.scope),
        score=float(result.score),
        bm25_score=float(result.bm25_score),
        vector_score=float(result.vector_score),
        graph_score=float(result.graph_score),
        trigger_score=float(result.trigger_score),
        methods=tuple(str(method) for method in result.methods),
    )


def run_item(service: MemoryRecallService, item: GoldsetItem) -> HarnessRun:
    """Replay one goldset item through the live recall path (read-only)."""

    results = service.memory_recall(
        item.query,
        scope=item.scope,
        ambient_context=item.ambient_context,
        depth=item.depth,
        max_results=item.max_results,
        log_access=False,
        log_event=False,
    )
    return HarnessRun(
        item=item,
        results=tuple(_harness_result(index, result) for index, result in enumerate(results)),
    )


def run_goldset(
    service: MemoryRecallService, items: Sequence[GoldsetItem]
) -> list[HarnessRun]:
    """Replay every item, always in ``query_id`` order."""

    return [run_item(service, item) for item in sorted(items, key=lambda item: item.query_id)]


# ---------------------------------------------------------------------------
# Query-anchor annotation (arm-independent)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnchorAnnotation:
    """What the anchor corpus holds about one goldset item.

    Derived from the snapshot and the item only -- never from the run -- so an
    anchors-on and an anchors-off arm over the same snapshot produce identical
    annotations and therefore identical bucket membership. Without that, the
    two arms would be scoring different item sets and the comparison would be
    meaningless.
    """

    query_id: str
    stratum: str
    scopes: tuple[str, ...]
    fingerprint: str
    exact_repeat: bool
    fingerprint_anchor_id: str | None
    matched_anchor_ids: tuple[str, ...]
    best_similarity: float | None
    nearest_similarity: float | None
    target_ids: tuple[str, ...]
    anchor_as_of: str | None
    fresh_relevant_ids: tuple[str, ...]

    @property
    def matched(self) -> bool:
        return bool(self.matched_anchor_ids)

    @property
    def in_fresh_slice(self) -> bool:
        return bool(self.matched_anchor_ids and self.target_ids and self.fresh_relevant_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "stratum": self.stratum,
            "scopes": list(self.scopes),
            "fingerprint": self.fingerprint,
            "exact_repeat": self.exact_repeat,
            "fingerprint_anchor_id": self.fingerprint_anchor_id,
            "matched_anchor_ids": list(self.matched_anchor_ids),
            "best_similarity": self.best_similarity,
            "nearest_similarity": self.nearest_similarity,
            "target_ids": list(self.target_ids),
            "anchor_as_of": self.anchor_as_of,
            "fresh_relevant_ids": list(self.fresh_relevant_ids),
            "in_fresh_slice": self.in_fresh_slice,
        }


def _node_created_at(store: MemoryStore, node_ids: Iterable[str]) -> dict[str, str]:
    """``{node_id: created_at}`` for the ids that still have a row."""

    ids = sorted(set(node_ids))
    found: dict[str, str] = {}
    connection = store.connection
    for start in range(0, len(ids), 900):
        chunk = ids[start : start + 900]
        placeholders = ",".join("?" * len(chunk))
        for row in connection.execute(
            f"SELECT id, created_at FROM nodes WHERE id IN ({placeholders})", chunk
        ):
            found[str(row["id"])] = str(row["created_at"] or "")
    return found


def annotate_anchors(
    service: MemoryRecallService, items: Sequence[GoldsetItem]
) -> dict[str, AnchorAnnotation]:
    """Annotate every item against the snapshot's anchor corpus.

    Uses the service's own encoder and scope resolver, and
    ``query_anchors.match_anchors`` at the shipped floor, so the annotation is
    the retrieval path's own view of the corpus. Deliberately independent of
    ``service.anchor_seeding``: the anchors-off arm must be annotated with the
    same anchors the anchors-on arm was offered, or the two arms would not be
    scoring the same buckets.

    Returns ``{}`` on a snapshot with no anchor table and on one with no live
    anchor -- the cold-start case, where every anchor bucket is empty rather
    than absent.
    """

    store = service.store
    # 0 on a pre-v7 file as well as on an empty corpus: both are "no anchors".
    if not store.count_query_anchors(include_decayed=False):
        return {}

    annotations: dict[str, AnchorAnnotation] = {}
    for item in sorted(items, key=lambda item: item.query_id):
        plan = service.scope_resolver.resolve(
            scope=item.scope,
            ambient_context=item.ambient_context,
            store=store,
        )
        plan_scopes = [normalize_scope(scope) for scope in plan.named_scopes]
        own_scope = normalize_scope(item.scope) if item.scope else (
            plan_scopes[0] if plan_scopes else "global"
        )
        # The item's own scope first: `fingerprint` below is the one the goal's
        # rule names, `recall_fingerprint(query, scope)`. The rest of the plan
        # follows because an anchor in any admitted scope is one the scan can
        # return, and an exact repeat found there is still a lookup, not
        # generalization.
        scopes = tuple(dict.fromkeys([own_scope, *plan_scopes]))
        embedding = service.embedder.embed(item.query)

        fingerprint = recall_fingerprint(item.query, own_scope)
        fingerprint_anchor_id: str | None = None
        for scope in scopes:
            found = store.find_query_anchor(scope, recall_fingerprint(item.query, scope))
            if found is not None and not found.decayed:
                fingerprint_anchor_id = found.id
                break

        matches = match_anchors(
            store,
            embedding,
            plan,
            limit=ANCHOR_MATCH_LIMIT,
            min_similarity=ANCHOR_MATCH_COSINE_THRESHOLD,
        )
        # The unfloored nearest neighbour is the diagnostic that separates "no
        # anchor exists for this class of query" from "one exists and the floor
        # rejected it"; the eval cannot read a zero match rate without it.
        nearest = match_anchors(
            store, embedding, plan, limit=1, min_similarity=-1.0, include_targets=False
        )

        target_ids: list[str] = []
        as_of: str | None = None
        for match in matches:
            stamp = str(match.anchor.updated_at or match.anchor.created_at or "")
            if stamp and (as_of is None or stamp > as_of):
                as_of = stamp
            for target_id, _weight in match.targets:
                if target_id not in target_ids:
                    target_ids.append(target_id)

        relevant = tuple(item.relevant_node_ids)
        created = _node_created_at(store, relevant) if (matches and as_of) else {}
        fresh = tuple(
            node_id
            for node_id in relevant
            if node_id not in set(target_ids)
            and (created.get(node_id) or "") > (as_of or "")
        )

        annotations[item.query_id] = AnchorAnnotation(
            query_id=item.query_id,
            stratum=item.stratum,
            scopes=scopes,
            fingerprint=fingerprint,
            exact_repeat=fingerprint_anchor_id is not None,
            fingerprint_anchor_id=fingerprint_anchor_id,
            matched_anchor_ids=tuple(match.anchor.id for match in matches),
            best_similarity=round(matches[0].similarity, 6) if matches else None,
            nearest_similarity=round(nearest[0].similarity, 6) if nearest else None,
            target_ids=tuple(sorted(target_ids)),
            anchor_as_of=as_of,
            fresh_relevant_ids=fresh,
        )
    return annotations


# ---------------------------------------------------------------------------
# Metrics (replay's accumulator, live scores)
# ---------------------------------------------------------------------------


def to_replay_event(
    run: HarnessRun, *, relevant_override: Sequence[str] | None = None
) -> ReplayEvent:
    """Adapt a live run into the shape ``MetricAccumulator`` consumes.

    The ``ReplayResult`` rows carry the **live** per-channel scores, and
    ``useful`` comes from the goldset's ``relevant_node_ids`` instead of
    replay's containment labeling — the goldset already made that judgement.

    Relevant nodes that retrieval never returned are appended as zero-scored
    placeholder results marked useful. ``MetricAccumulator`` reads
    ``events_with_useful`` off ``event.useful_ids``, which is derived from the
    result list, so without the placeholders a total miss would drop out of the
    denominator entirely and read as "not evaluated" instead of "missed" —
    hiding exactly the recall failures phase 1 is meant to fix. Placeholders
    are never put into ``order`` / ``zero_order``, so they contribute a miss
    (no hit, no reciprocal rank) and nothing else.

    ``relevant_override`` narrows the useful set without touching the ranking:
    the fresh-node-visibility slice scores the same returned list against the
    fresh relevant nodes only. An empty override yields an event with no useful
    result, which ``MetricAccumulator`` drops from ``events_with_useful``, so an
    item is never scored against a relevance set it does not have.
    """

    relevant = set(
        run.item.relevant_node_ids if relevant_override is None else relevant_override
    )
    scope = run.item.scope or "global"
    results = [
        ReplayResult(
            node_id=result.node_id,
            rank=result.rank,
            level=result.level if result.level in NODE_LEVELS else "trace",
            scope=result.scope,
            score=result.score,
            bm25_score=result.bm25_score,
            vector_score=result.vector_score,
            graph_score=result.graph_score,
            trigger_score=result.trigger_score,
            path=(),
            useful=result.node_id in relevant,
        )
        for result in run.results
    ]
    returned = {result.node_id for result in run.results}
    for offset, node_id in enumerate(sorted(relevant - returned)):
        results.append(
            ReplayResult(
                node_id=node_id,
                rank=len(run.results) + offset + 1,
                level="trace",
                scope=scope,
                score=0.0,
                bm25_score=0.0,
                vector_score=0.0,
                graph_score=0.0,
                trigger_score=0.0,
                path=(),
                useful=True,
            )
        )
    return ReplayEvent(
        id=run.item.query_id,
        created_at="",
        scope=scope,
        requested_scope=scope,
        resolved_scopes=(scope,),
        depth=None if run.item.depth is None else str(run.item.depth),
        max_results=run.item.max_results,
        query=run.item.query,
        # Never None: `ReplayEvent.labeled` gates nothing here, but keeping it
        # truthful marks these events as judged.
        feedback_trace_id=run.item.source_event_id or run.item.query_id,
        feedback_applied_at=None,
        results=results,
    )


def live_order(run: HarnessRun) -> list[str]:
    return [result.node_id for result in run.results]


def zero_graph_order(run: HarnessRun) -> list[str]:
    """Live order minus results whose only evidence is the graph channel."""

    return [
        result.node_id for result in run.results if tuple(result.methods) != ("graph",)
    ]


def channel_score(result: HarnessResult, channel: str) -> float:
    return float(getattr(result, f"{channel}_score"))


def channel_order(run: HarnessRun, channel: str) -> list[str]:
    """Same results, re-ordered by one channel score; ties keep the live rank."""

    ordered = sorted(
        run.results, key=lambda result: (-channel_score(result, channel), result.rank)
    )
    return [result.node_id for result in ordered]


_CHANNEL_BLOCK_KEYS = ("events", "events_with_useful", "hit@1", "hit@5", "hit@10", "mrr")


def _channel_block(accumulator: MetricAccumulator) -> dict[str, Any]:
    block = accumulator.to_dict()
    return {key: block[key] for key in _CHANNEL_BLOCK_KEYS}


class _Attribution:
    """First-relevant-result method counts, plus the exact method *sets*.

    ``methods`` counts a result found by two channels once for each, which is
    what the phase-1 report has always shown. ``method_sets`` keeps the
    combination intact (``graph+trigger`` distinct from ``trigger``), which is
    the only way to read one channel's *unique* contribution: retiring a
    channel costs the hits nothing else found, not its whole share.
    """

    def __init__(self) -> None:
        self.items = 0
        self.methods: Counter[str] = Counter()
        self.method_sets: Counter[str] = Counter()

    def add(self, methods: Sequence[str]) -> None:
        self.items += 1
        self.methods.update(methods)
        self.method_sets["+".join(sorted(set(methods))) or "none"] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "items_with_relevant_hit": self.items,
            "methods": {channel: self.methods.get(channel, 0) for channel in CHANNELS},
            "method_share": {
                channel: _ratio(self.methods.get(channel, 0), self.items)
                for channel in CHANNELS
            },
            "method_sets": dict(sorted(self.method_sets.items())),
            "unique_to_channel": {
                channel: self.method_sets.get(channel, 0) for channel in CHANNELS
            },
            "unique_share": {
                channel: _ratio(self.method_sets.get(channel, 0), self.items)
                for channel in CHANNELS
            },
        }


def _percentiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def at(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        return round(ordered[index], 6)

    return {
        "min": round(ordered[0], 6),
        "p50": at(0.50),
        "p90": at(0.90),
        "p99": at(0.99),
        "max": round(ordered[-1], 6),
    }


def compute_metrics(
    runs: Sequence[HarnessRun],
    annotations: Mapping[str, AnchorAnnotation] | None = None,
) -> dict[str, Any]:
    """Overall / per-stratum / per-channel / tail metrics for a set of runs.

    With ``annotations`` (see :func:`annotate_anchors`) three anchor buckets
    join them as first-class metrics: ``anchor_holdout`` (exact repeat versus
    fingerprint-disjoint), ``fresh_node_visibility`` (the self-reinforcement
    guard) and ``anchor_coverage``. Because the annotation is derived from the
    snapshot rather than from the run, the same annotations apply to an
    anchors-on and an anchors-off arm, and the buckets compare like for like.
    """

    annotations = dict(annotations or {})
    overall = MetricAccumulator()
    tail = MetricAccumulator()
    per_stratum: dict[str, MetricAccumulator] = {}
    per_channel = {channel: MetricAccumulator() for channel in CHANNELS}
    attribution = _Attribution()
    per_stratum_attribution: dict[str, _Attribution] = {}
    fresh = MetricAccumulator()
    holdout = {"exact_repeat": MetricAccumulator(), "fingerprint_disjoint": MetricAccumulator()}
    holdout_per_stratum: dict[str, dict[str, MetricAccumulator]] = {}
    fresh_relevant_nodes = 0
    nearest_similarities: list[float] = []
    coverage: Counter[str] = Counter()
    coverage_per_stratum: dict[str, Counter[str]] = {}

    for run in sorted(runs, key=lambda run: run.item.query_id):
        stratum = run.item.stratum
        event = to_replay_event(run)
        order = live_order(run)
        zero_order = zero_graph_order(run)
        overall.add_event(event, order, zero_order)
        per_stratum.setdefault(stratum, MetricAccumulator()).add_event(
            event, order, zero_order
        )
        if run.item.tail:
            tail.add_event(event, order, zero_order)
        for channel, accumulator in per_channel.items():
            single = channel_order(run, channel)
            accumulator.add_event(event, single, single)
        relevant = set(run.item.relevant_node_ids)
        first = next(
            (result for result in run.results if result.node_id in relevant), None
        )
        if first is not None:
            attribution.add(first.methods)
            per_stratum_attribution.setdefault(stratum, _Attribution()).add(first.methods)

        annotation = annotations.get(run.item.query_id)
        if annotation is None:
            continue
        bucket = "exact_repeat" if annotation.exact_repeat else "fingerprint_disjoint"
        holdout[bucket].add_event(event, order, zero_order)
        holdout_per_stratum.setdefault(
            stratum, {key: MetricAccumulator() for key in holdout}
        )[bucket].add_event(event, order, zero_order)

        stratum_coverage = coverage_per_stratum.setdefault(stratum, Counter())
        for counter in (coverage, stratum_coverage):
            counter["items"] += 1
            counter["exact_repeat"] += int(annotation.exact_repeat)
            counter["fingerprint_disjoint"] += int(not annotation.exact_repeat)
            counter["matched"] += int(annotation.matched)
            counter["matched_with_live_targets"] += int(
                bool(annotation.matched_anchor_ids and annotation.target_ids)
            )
            counter["fresh_slice"] += int(annotation.in_fresh_slice)
        if annotation.nearest_similarity is not None:
            nearest_similarities.append(annotation.nearest_similarity)
        if annotation.in_fresh_slice:
            fresh_relevant_nodes += len(annotation.fresh_relevant_ids)
            fresh.add_event(
                to_replay_event(run, relevant_override=annotation.fresh_relevant_ids),
                order,
                zero_order,
            )

    def _coverage_block(counter: Counter[str]) -> dict[str, Any]:
        items = counter["items"]
        return {
            "items": items,
            "exact_repeat": counter["exact_repeat"],
            "fingerprint_disjoint": counter["fingerprint_disjoint"],
            "matched": counter["matched"],
            "matched_share": _ratio(counter["matched"], items),
            "matched_with_live_targets": counter["matched_with_live_targets"],
            "fresh_slice_items": counter["fresh_slice"],
        }

    return {
        "harness_version": HARNESS_VERSION,
        "hit_ks": list(HIT_KS),
        "items": len(runs),
        "tail_items": sum(1 for run in runs if run.item.tail),
        "overall": overall.to_dict(),
        "per_stratum": {
            stratum: accumulator.to_dict()
            for stratum, accumulator in sorted(per_stratum.items())
        },
        "tail": tail.to_dict(),
        "per_channel": {
            channel: _channel_block(accumulator)
            for channel, accumulator in sorted(per_channel.items())
        },
        "channel_attribution": {
            **attribution.to_dict(),
            "per_stratum": {
                stratum: block.to_dict()
                for stratum, block in sorted(per_stratum_attribution.items())
            },
        },
        "anchor_holdout": {
            "annotated_items": coverage["items"],
            **{key: accumulator.to_dict() for key, accumulator in sorted(holdout.items())},
            "per_stratum": {
                stratum: {
                    key: accumulator.to_dict()
                    for key, accumulator in sorted(buckets.items())
                }
                for stratum, buckets in sorted(holdout_per_stratum.items())
            },
        },
        "fresh_node_visibility": {
            "items": coverage["fresh_slice"],
            "fresh_relevant_nodes": fresh_relevant_nodes,
            **fresh.to_dict(),
        },
        "anchor_coverage": {
            **_coverage_block(coverage),
            "nearest_similarity": _percentiles(nearest_similarities),
            "match_floor": ANCHOR_MATCH_COSINE_THRESHOLD,
            "match_limit": ANCHOR_MATCH_LIMIT,
            "per_stratum": {
                stratum: _coverage_block(counter)
                for stratum, counter in sorted(coverage_per_stratum.items())
            },
        },
    }


# ---------------------------------------------------------------------------
# Live-agreement sanity check
# ---------------------------------------------------------------------------


def seeded_sample(values: Iterable[str], size: int, seed: int) -> list[str]:
    """Deterministic sample over the sorted, de-duplicated id list."""

    pool = sorted(set(values))
    if size <= 0:
        return []
    if size >= len(pool):
        return pool
    return sorted(random.Random(seed).sample(pool, size))


def live_agreement(
    working_db: Path,
    items: Sequence[GoldsetItem],
    runs: Sequence[HarnessRun],
    *,
    sample_size: int,
    seed: int,
    top_k: int = AGREEMENT_TOP_K,
    anchor_seeding: bool = True,
) -> dict[str, Any]:
    """Compare the runner's top-k with an independently built live service.

    Deliberately does *not* call :func:`run_item`: a fresh ``MemoryStore`` +
    ``MemoryRecallService`` pair is constructed here on the same working copy
    and driven directly, so a bug in the runner's own plumbing cannot hide.
    ``anchor_seeding`` must mirror the runner's, or the check would compare an
    anchors-off arm against an anchors-on service and report the ablation as a
    divergence.
    """

    by_id = {item.query_id: item for item in items}
    runs_by_id = {run.item.query_id: run for run in runs}
    sampled = seeded_sample(runs_by_id, sample_size, seed)
    divergences: list[dict[str, Any]] = []
    if sampled:
        store = MemoryStore(MemoryConfig(db_path=working_db))
        try:
            service = MemoryRecallService(store, anchor_seeding=anchor_seeding)
            for query_id in sampled:
                item = by_id[query_id]
                results = service.memory_recall(
                    item.query,
                    scope=item.scope,
                    ambient_context=item.ambient_context,
                    depth=item.depth,
                    max_results=item.max_results,
                    log_access=False,
                    log_event=False,
                )
                direct = [result.node_id for result in results][:top_k]
                harness = list(runs_by_id[query_id].node_ids)[:top_k]
                if direct != harness:
                    divergences.append(
                        {
                            "query_id": query_id,
                            "query": item.query,
                            "harness_top_k": harness,
                            "direct_top_k": direct,
                        }
                    )
        finally:
            store.close()
    return {
        "definition": AGREEMENT_RULE,
        "seed": seed,
        "top_k": top_k,
        "requested_sample": sample_size,
        "sampled_items": len(sampled),
        "sampled_query_ids": sampled,
        "agreed_items": len(sampled) - len(divergences),
        "agreement_rate": _ratio(len(sampled) - len(divergences), len(sampled))
        if sampled
        else None,
        "divergences": divergences,
    }


# ---------------------------------------------------------------------------
# Reporters
# ---------------------------------------------------------------------------


def git_commit(cwd: str | Path | None = None) -> str | None:
    directory = Path(cwd) if cwd is not None else Path(__file__).resolve().parent
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(directory),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment guard
        return None
    commit = completed.stdout.strip()
    return commit or None


def embedding_backend() -> str:
    return os.environ.get("LIVING_MEMORY_EMBEDDING_BACKEND") or "auto"


def build_report(
    *,
    runs: Sequence[HarnessRun],
    metrics: Mapping[str, Any],
    agreement: Mapping[str, Any],
    manifest: Mapping[str, Any],
    goldset_path: Path,
    goldset_items: int,
    cutoff: str | None,
    seed: int,
    divergence_note: str | None,
    generated_at: str | None = None,
    annotations: Mapping[str, AnchorAnnotation] | None = None,
    anchor_seeding: bool = True,
    anchor_corpus: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the JSON report: byte-stable ``metrics``, dated ``provenance``."""

    return {
        "metrics": dict(metrics),
        "agreement": dict(agreement),
        "runs": [run.to_dict() for run in sorted(runs, key=lambda run: run.item.query_id)],
        "anchor_annotations": [
            annotation.to_dict()
            for _query_id, annotation in sorted((annotations or {}).items())
        ],
        "definitions": {
            "tail_rule": TAIL_RULE,
            "per_channel": PER_CHANNEL_RULE,
            "live_agreement": AGREEMENT_RULE,
            "anchor_slice": ANCHOR_SLICE_RULE,
        },
        "provenance": {
            "generated_at": generated_at or utc_now_iso(),
            "harness_version": HARNESS_VERSION,
            "source_db_path": manifest.get("source_path"),
            "snapshot_path": manifest.get("snapshot_path"),
            "snapshot_sha256": manifest.get("snapshot_sha256"),
            "snapshot_captured_at": manifest.get("captured_at"),
            "snapshot_row_counts": manifest.get("row_counts", {}),
            "recall_events_created_at": manifest.get("recall_events_created_at", {}),
            "cutoff": cutoff,
            "seed": seed,
            "git_commit": git_commit(),
            "goldset_path": str(goldset_path),
            "goldset_sha256": sha256_file(goldset_path),
            "goldset_items": goldset_items,
            "embedding_backend": embedding_backend(),
            "embedding_model": MemoryConfig().embedding_model,
            "divergence_note": divergence_note,
            # The one bit that separates the two arms of the holdout run.
            "anchor_seeding": bool(anchor_seeding),
            "anchor_corpus": dict(anchor_corpus or {}),
        },
    }


def _metric_row(label: str, block: Mapping[str, Any]) -> str:
    return (
        f"| {label} | {block['events']} | {block['events_with_useful']} "
        f"| {block['hit@1']:.3f} | {block['hit@5']:.3f} | {block['hit@10']:.3f} "
        f"| {block['mrr']:.3f} |"
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    """Human-readable companion to the JSON report.

    ``replay.render_markdown`` cannot be reused directly: it indexes
    ``corpus`` / ``split`` / ``labeling`` / ``graph_zeroed`` keys this report
    does not have. Its table and section idiom is mirrored instead of forked.
    """

    metrics = report["metrics"]
    provenance = report["provenance"]
    agreement = report["agreement"]
    counts = provenance.get("snapshot_row_counts") or {}

    lines: list[str] = []
    lines.append("# End-to-end retrieval-harness baseline")
    lines.append("")
    lines.append(
        f"Generated: {provenance['generated_at']} | snapshot "
        f"`{provenance.get('snapshot_path')}` | commit `{provenance.get('git_commit')}`"
    )
    lines.append("")
    lines.append("## Snapshot and goldset")
    lines.append("")
    lines.append(f"- Source DB: `{provenance.get('source_db_path')}`")
    lines.append(
        f"- Snapshot sha256 `{provenance.get('snapshot_sha256')}` captured "
        f"{provenance.get('snapshot_captured_at')}"
    )
    lines.append(
        "- Snapshot rows: "
        + ", ".join(f"{name} {counts.get(name)}" for name in sorted(counts))
    )
    span = provenance.get("recall_events_created_at") or {}
    lines.append(
        f"- recall_events span {span.get('min')} .. {span.get('max')} | "
        f"cutoff `{provenance.get('cutoff')}` | seed {provenance.get('seed')}"
    )
    lines.append(
        f"- Goldset `{provenance.get('goldset_path')}` sha256 "
        f"`{provenance.get('goldset_sha256')}` with {provenance.get('goldset_items')} items "
        f"({metrics['tail_items']} tail)"
    )
    lines.append(
        f"- Embedding backend `{provenance.get('embedding_backend')}` "
        f"model `{provenance.get('embedding_model')}` | harness v{provenance['harness_version']}"
    )
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| bucket | items | w/relevant | hit@1 | hit@5 | hit@10 | MRR |")
    lines.append("|" + "---|" * 7)
    lines.append(_metric_row("overall", metrics["overall"]))
    for stratum, block in metrics["per_stratum"].items():
        lines.append(_metric_row(f"stratum:{stratum}", block))
    lines.append(_metric_row("tail", metrics["tail"]))
    if "fresh_node_visibility" in metrics:
        lines.append(_metric_row("fresh_node_visibility", metrics["fresh_node_visibility"]))
    lines.append("")
    holdout = metrics.get("anchor_holdout") or {}
    coverage = metrics.get("anchor_coverage") or {}
    if holdout.get("annotated_items"):
        lines.append(
            "### Anchor holdout "
            f"(anchor_seeding={provenance.get('anchor_seeding')}, "
            f"{(provenance.get('anchor_corpus') or {}).get('live_anchors')} live anchors)"
        )
        lines.append("")
        lines.append("| bucket | items | w/relevant | hit@1 | hit@5 | hit@10 | MRR |")
        lines.append("|" + "---|" * 7)
        for key in ("exact_repeat", "fingerprint_disjoint"):
            lines.append(_metric_row(key, holdout[key]))
        for stratum, buckets in holdout.get("per_stratum", {}).items():
            for key in ("exact_repeat", "fingerprint_disjoint"):
                lines.append(_metric_row(f"{stratum}:{key}", buckets[key]))
        lines.append("")
        lines.append("### Anchor coverage of the goldset")
        lines.append("")
        lines.append(
            "| bucket | items | exact repeat | disjoint | matched | matched share | "
            "matched w/ live targets | fresh-node slice |"
        )
        lines.append("|" + "---|" * 8)

        def coverage_row(label: str, block: Mapping[str, Any]) -> str:
            return (
                f"| {label} | {block['items']} | {block['exact_repeat']} "
                f"| {block['fingerprint_disjoint']} | {block['matched']} "
                f"| {block['matched_share']:.3f} | {block['matched_with_live_targets']} "
                f"| {block['fresh_slice_items']} |"
            )

        lines.append(coverage_row("all", coverage))
        for stratum, block in coverage.get("per_stratum", {}).items():
            lines.append(coverage_row(stratum, block))
        nearest = coverage.get("nearest_similarity")
        if nearest:
            lines.append("")
            lines.append(
                f"- Nearest in-scope anchor cosine, floor {coverage.get('match_floor')}: "
                + ", ".join(f"{name} {nearest[name]:.3f}" for name in sorted(nearest))
            )
        lines.append("")
    lines.append("### Per channel")
    lines.append("")
    lines.append("| channel | items | w/relevant | hit@1 | hit@5 | hit@10 | MRR |")
    lines.append("|" + "---|" * 7)
    for channel, block in metrics["per_channel"].items():
        lines.append(_metric_row(channel, block))
    lines.append("")
    attribution = metrics["channel_attribution"]
    lines.append(
        "### Channel attribution "
        f"({attribution['items_with_relevant_hit']} items with a relevant result)"
    )
    lines.append("")
    lines.append("| method | first-relevant results | share | unique to it | unique share |")
    lines.append("|" + "---|" * 5)
    for channel in CHANNELS:
        lines.append(
            f"| {channel} | {attribution['methods'][channel]} "
            f"| {attribution['method_share'][channel]:.3f} "
            f"| {attribution.get('unique_to_channel', {}).get(channel, 0)} "
            f"| {attribution.get('unique_share', {}).get(channel, 0.0):.3f} |"
        )
    lines.append("")
    for stratum, block in (attribution.get("per_stratum") or {}).items():
        lines.append(
            f"- `{stratum}` ({block['items_with_relevant_hit']} items with a relevant "
            "result), method sets: "
            + ", ".join(
                f"`{name}` {count}" for name, count in block["method_sets"].items()
            )
        )
    lines.append("")
    lines.append("## Live-agreement sanity")
    lines.append("")
    rate = agreement.get("agreement_rate")
    lines.append(
        f"- Sampled {agreement['sampled_items']} of {provenance.get('goldset_items')} items "
        f"(seed {agreement['seed']}, top-{agreement['top_k']}): "
        f"agreement_rate {'n/a' if rate is None else f'{rate:.3f}'}"
    )
    if agreement["divergences"]:
        lines.append("")
        lines.append("| query_id | harness top-k | direct top-k |")
        lines.append("|" + "---|" * 3)
        for divergence in agreement["divergences"]:
            lines.append(
                f"| {divergence['query_id']} | {', '.join(divergence['harness_top_k'])} "
                f"| {', '.join(divergence['direct_top_k'])} |"
            )
    else:
        lines.append("- No divergences.")
    if provenance.get("divergence_note"):
        lines.append("")
        lines.append(f"- Divergence note: {provenance['divergence_note']}")
    lines.append("")
    lines.append("## Definitions")
    lines.append("")
    lines.append("### Tail rule")
    lines.append("")
    lines.append("```text")
    lines.append(TAIL_RULE)
    lines.append("```")
    lines.append("")
    lines.append("### Per-channel buckets and channel attribution")
    lines.append("")
    lines.append("```text")
    lines.append(PER_CHANNEL_RULE)
    lines.append("```")
    lines.append("")
    lines.append("### Live agreement")
    lines.append("")
    lines.append("```text")
    lines.append(AGREEMENT_RULE)
    lines.append("```")
    lines.append("")
    lines.append("### Anchor holdout, coverage and fresh-node visibility")
    lines.append("")
    lines.append("```text")
    lines.append(ANCHOR_SLICE_RULE)
    lines.append("```")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Relevance is the goldset's judgement, frozen at build time: "
        "`content_grounded` items inherit replay's IDF-containment label over the "
        "consuming trace, `cross_lingual` items inherit the paraphrase query's own "
        "top-k (so they measure jargon-vs-paraphrase agreement, not absolute truth), "
        "and `role_query` items are curated against active `level:schema` triggers."
    )
    lines.append(
        "- A relevant node that retrieval never returns counts as a **miss**, not as "
        "an unevaluated item: it enters the denominator with no hit and no reciprocal "
        "rank. The run therefore measures ranking and candidate collection together "
        "and cannot tell one failure from the other."
    )
    lines.append(
        "- Per-channel buckets re-order only what retrieval already returned, so they "
        "bound a channel's ranking power, not its recall."
    )
    lines.append("")
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], report_path: Path, markdown_path: Path | None) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")


# ---------------------------------------------------------------------------
# Goldset construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GoldsetBuildConfig:
    """Everything ``build-goldset`` needs besides the snapshot itself."""

    cutoff: str
    seed: int = 0
    seed_queries: Path | None = None
    content_grounded_cap: int | None = None
    cross_lingual_cap: int | None = None
    role_query_cap: int | None = None
    cross_lingual_top_k: int = 5
    min_containment: float = LabelConfig().min_containment

    @property
    def label(self) -> LabelConfig:
        return LabelConfig(min_containment=self.min_containment)


def _stable_id(stratum: str, seed_text: str) -> str:
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:12]
    return f"{stratum}-{digest}"


def _fetch_node_rows(
    connection: sqlite3.Connection, node_ids: Iterable[str]
) -> dict[str, sqlite3.Row]:
    ids = sorted(set(node_ids))
    rows: dict[str, sqlite3.Row] = {}
    for start in range(0, len(ids), 900):
        chunk = ids[start : start + 900]
        placeholders = ",".join("?" * len(chunk))
        for row in connection.execute(
            f"SELECT id, content, level, decayed, context FROM nodes WHERE id IN ({placeholders})",
            chunk,
        ):
            rows[str(row["id"])] = row
    return rows


def _active_ids(rows: Mapping[str, sqlite3.Row]) -> set[str]:
    return {node_id for node_id, row in rows.items() if not int(row["decayed"])}


def _event_extras(connection: sqlite3.Connection, event_ids: Sequence[str]) -> dict[str, Any]:
    """``ambient_context`` per event id (``ReplayEvent`` does not carry it)."""

    extras: dict[str, Any] = {}
    ids = sorted(set(event_ids))
    for start in range(0, len(ids), 900):
        chunk = ids[start : start + 900]
        placeholders = ",".join("?" * len(chunk))
        for row in connection.execute(
            f"SELECT id, ambient_context FROM recall_events WHERE id IN ({placeholders})",
            chunk,
        ):
            try:
                ambient = json.loads(row["ambient_context"] or "{}")
            except ValueError:
                ambient = {}
            extras[str(row["id"])] = ambient if isinstance(ambient, dict) else {}
    return extras


def build_content_grounded(
    connection: sqlite3.Connection,
    config: GoldsetBuildConfig,
    *,
    tokenizer: VisibleSpanTokenizer,
    build_stamp: Mapping[str, Any],
) -> tuple[list[GoldsetItem], dict[str, Any]]:
    """Sample consumed recall events labeled useful by replay's IDF containment."""

    cutoff = normalize_cutoff(config.cutoff)
    events, load_stats = load_replay_events(connection)
    candidates = [
        event for event in events if event.labeled and event.created_at >= cutoff
    ]
    candidates.sort(key=lambda event: (event.created_at, event.id))
    label_data = build_label_data(connection, candidates)
    threshold = config.label.min_containment

    node_rows = _fetch_node_rows(
        connection,
        (result.node_id for event in candidates for result in event.results),
    )
    active = _active_ids(node_rows)

    eligible: dict[str, tuple[Any, tuple[str, ...], dict[str, float]]] = {}
    dropped_no_useful = 0
    dropped_inactive = 0
    for event in candidates:
        containments = {
            result.node_id: _containment(label_data, result.node_id, event.feedback_trace_id)
            for result in event.results
        }
        useful = tuple(
            sorted(
                node_id
                for node_id, value in containments.items()
                if value >= threshold
            )
        )
        if not useful:
            dropped_no_useful += 1
            continue
        if any(node_id not in active for node_id in useful):
            dropped_inactive += 1
            continue
        eligible[event.id] = (event, useful, containments)

    ordered_ids = sorted(eligible)
    cap = config.content_grounded_cap
    if cap is not None and 0 <= cap < len(ordered_ids):
        chosen = sorted(random.Random(config.seed).sample(ordered_ids, cap))
    else:
        chosen = ordered_ids

    extras = _event_extras(connection, chosen)
    trace_ids = [
        eligible[event_id][0].feedback_trace_id
        for event_id in chosen
        if eligible[event_id][0].feedback_trace_id
    ]
    trace_rows = _fetch_node_rows(connection, trace_ids)
    contents = {node_id: str(row["content"]) for node_id, row in node_rows.items()}

    items: list[GoldsetItem] = []
    for event_id in chosen:
        event, useful, containments = eligible[event_id]
        trace_row = trace_rows.get(event.feedback_trace_id or "")
        justifying_text = str(trace_row["content"]) if trace_row is not None else ""
        tail, tail_audit = classify_item_tail(
            contents, useful, justifying_text, tokenizer=tokenizer
        )
        items.append(
            GoldsetItem(
                query_id=_stable_id("content_grounded", event.id),
                query=event.query,
                scope=event.requested_scope,
                ambient_context=extras.get(event.id) or None,
                depth=event.depth,
                max_results=max(event.max_results, MIN_MAX_RESULTS),
                relevant_node_ids=useful,
                stratum="content_grounded",
                tail=tail,
                source_event_id=event.id,
                provenance={
                    "build": dict(build_stamp),
                    "source_max_results": event.max_results,
                    "recorded_scope": event.scope,
                    "requested_scope": event.requested_scope,
                    "resolved_scopes": list(event.resolved_scopes),
                    "recorded_created_at": event.created_at,
                    "feedback_trace_id": event.feedback_trace_id,
                    "label": config.label.to_dict(),
                    "containment": {
                        node_id: round(value, 6)
                        for node_id, value in sorted(containments.items())
                    },
                    "justifying_text": "consuming_trace_content",
                    "tail_classification": tail_audit,
                },
            )
        )

    stats = {
        "loader": load_stats,
        "cutoff": cutoff,
        "candidate_events": len(candidates),
        "eligible_events": len(ordered_ids),
        "dropped_no_useful_result": dropped_no_useful,
        "dropped_inactive_node": dropped_inactive,
        "selected": len(items),
        "cap": cap,
    }
    return items, stats


def _seed_specs(path: Path | None, key: str) -> list[dict[str, Any]]:
    if path is None:
        return []
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise GoldsetError(f"seed-queries {path}: expected a JSON object at the top level")
    specs = loaded.get(key, [])
    if not isinstance(specs, list):
        raise GoldsetError(f"seed-queries {path}: '{key}' must be a list")
    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise GoldsetError(f"seed-queries {path}: {key}[{index}] must be an object")
    return [dict(spec) for spec in specs]


def _require_text(spec: Mapping[str, Any], key: str, label: str) -> str:
    value = spec.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GoldsetError(f"{label}: '{key}' must be a non-empty string")
    return value


def build_cross_lingual(
    service: MemoryRecallService,
    specs: Sequence[Mapping[str, Any]],
    config: GoldsetBuildConfig,
    *,
    tokenizer: VisibleSpanTokenizer,
    build_stamp: Mapping[str, Any],
) -> tuple[list[GoldsetItem], dict[str, Any]]:
    """Resolve jargon queries mechanically through their paraphrase's top-k."""

    prepared: list[tuple[str, Mapping[str, Any]]] = []
    for index, spec in enumerate(specs):
        label = f"cross_lingual spec #{index}"
        jargon = _require_text(spec, "jargon_query", label)
        _require_text(spec, "paraphrase_query", label)
        query_id = spec.get("query_id") or _stable_id("cross_lingual", jargon)
        if not isinstance(query_id, str) or not query_id.strip():
            raise GoldsetError(f"{label}: 'query_id' must be a non-empty string")
        prepared.append((query_id, spec))
    prepared.sort(key=lambda entry: entry[0])
    cap = config.cross_lingual_cap
    selected = prepared if cap is None or cap < 0 else prepared[:cap]

    items: list[GoldsetItem] = []
    for query_id, spec in selected:
        jargon = str(spec["jargon_query"])
        paraphrase = str(spec["paraphrase_query"])
        scope = spec.get("scope")
        depth = spec.get("depth", 1)
        top_k = int(spec.get("top_k") or config.cross_lingual_top_k)
        paraphrase_results = service.memory_recall(
            paraphrase,
            scope=scope,
            ambient_context=None,
            depth=depth,
            max_results=top_k,
            log_access=False,
            log_event=False,
        )
        if not paraphrase_results:
            raise GoldsetError(
                f"cross_lingual {query_id}: the paraphrase query returned no results, "
                "so relevance cannot be resolved mechanically"
            )
        relevant = tuple(result.node_id for result in paraphrase_results)
        jargon_vector = service.embedder.embed(jargon)
        contents: dict[str, str] = {}
        jargon_vector_scores: dict[str, float | None] = {}
        for result in paraphrase_results:
            contents[result.node_id] = result.node.content
            embedding = result.node.embedding
            jargon_vector_scores[result.node_id] = (
                round(cosine_similarity(jargon_vector, embedding), 6)
                if embedding
                else None
            )
        tail, tail_audit = classify_item_tail(
            contents, relevant, jargon, tokenizer=tokenizer
        )
        items.append(
            GoldsetItem(
                query_id=query_id,
                query=jargon,
                scope=scope if isinstance(scope, str) else None,
                ambient_context=None,
                depth=depth,
                max_results=max(int(spec.get("max_results") or MIN_MAX_RESULTS), MIN_MAX_RESULTS),
                relevant_node_ids=relevant,
                stratum="cross_lingual",
                tail=tail,
                source_event_id=None,
                provenance={
                    "build": dict(build_stamp),
                    "jargon_query": jargon,
                    "paraphrase_query": paraphrase,
                    "resolution": "paraphrase_top_k",
                    "paraphrase_top_k": [
                        {
                            "node_id": result.node_id,
                            "rank": rank + 1,
                            "score": result.score,
                            "bm25_score": result.bm25_score,
                            "vector_score": result.vector_score,
                            "graph_score": result.graph_score,
                            "trigger_score": result.trigger_score,
                            "methods": list(result.methods),
                        }
                        for rank, result in enumerate(paraphrase_results)
                    ],
                    "jargon_vector_scores": dict(sorted(jargon_vector_scores.items())),
                    "justifying_text": "goldset_query",
                    "tail_classification": tail_audit,
                },
            )
        )
    return items, {"specs": len(prepared), "selected": len(items), "cap": cap}


def build_role_query(
    store: MemoryStore,
    specs: Sequence[Mapping[str, Any]],
    config: GoldsetBuildConfig,
    *,
    tokenizer: VisibleSpanTokenizer,
    build_stamp: Mapping[str, Any],
) -> tuple[list[GoldsetItem], dict[str, Any]]:
    """Curated role/procedural queries against active ``level:schema`` triggers."""

    prepared: list[tuple[str, Mapping[str, Any]]] = []
    for index, spec in enumerate(specs):
        label = f"role_query spec #{index}"
        query = _require_text(spec, "query", label)
        query_id = spec.get("query_id") or _stable_id("role_query", query)
        if not isinstance(query_id, str) or not query_id.strip():
            raise GoldsetError(f"{label}: 'query_id' must be a non-empty string")
        prepared.append((query_id, spec))
    prepared.sort(key=lambda entry: entry[0])
    cap = config.role_query_cap
    selected = prepared if cap is None or cap < 0 else prepared[:cap]

    items: list[GoldsetItem] = []
    for query_id, spec in selected:
        query = str(spec["query"])
        raw_ids = spec.get("expected_node_ids") or spec.get("relevant_node_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise GoldsetError(
                f"role_query {query_id}: 'expected_node_ids' must be a non-empty list"
            )
        expected = tuple(sorted({str(node_id) for node_id in raw_ids}))
        triggers: dict[str, str] = {}
        contents: dict[str, str] = {}
        for node_id in expected:
            node = store.get_node(node_id)
            if node is None:
                raise GoldsetError(f"role_query {query_id}: node {node_id} is not in the snapshot")
            if node.decayed:
                raise GoldsetError(f"role_query {query_id}: node {node_id} is decayed (inactive)")
            if node.level != "schema":
                raise GoldsetError(
                    f"role_query {query_id}: node {node_id} has level "
                    f"{node.level!r}, expected 'schema'"
                )
            trigger = node.context.get("trigger")
            if not isinstance(trigger, str) or not trigger.strip():
                raise GoldsetError(
                    f"role_query {query_id}: node {node_id} carries no context.trigger"
                )
            triggers[node_id] = trigger
            contents[node_id] = node.content
        tail, tail_audit = classify_item_tail(contents, expected, query, tokenizer=tokenizer)
        scope = spec.get("scope")
        items.append(
            GoldsetItem(
                query_id=query_id,
                query=query,
                scope=scope if isinstance(scope, str) else None,
                ambient_context=None,
                depth=spec.get("depth", 1),
                max_results=max(int(spec.get("max_results") or MIN_MAX_RESULTS), MIN_MAX_RESULTS),
                relevant_node_ids=expected,
                stratum="role_query",
                tail=tail,
                source_event_id=None,
                provenance={
                    "build": dict(build_stamp),
                    "resolution": "curated_schema_triggers",
                    "justification_triggers": dict(sorted(triggers.items())),
                    "justifying_text": "goldset_query",
                    "tail_classification": tail_audit,
                },
            )
        )
    return items, {"specs": len(prepared), "selected": len(items), "cap": cap}


def build_goldset(
    working_db: Path,
    config: GoldsetBuildConfig,
    *,
    tokenizer: VisibleSpanTokenizer,
    build_stamp: Mapping[str, Any] | None = None,
) -> tuple[list[GoldsetItem], dict[str, Any]]:
    """Build all three strata against one working copy of the snapshot."""

    stamp = dict(build_stamp or {})
    stamp.setdefault("cutoff", normalize_cutoff(config.cutoff))
    stamp.setdefault("seed", config.seed)
    stamp.setdefault("harness_version", HARNESS_VERSION)

    store = MemoryStore(MemoryConfig(db_path=working_db))
    try:
        service = MemoryRecallService(store)
        content_items, content_stats = build_content_grounded(
            store.connection, config, tokenizer=tokenizer, build_stamp=stamp
        )
        cross_items, cross_stats = build_cross_lingual(
            service,
            _seed_specs(config.seed_queries, "cross_lingual"),
            config,
            tokenizer=tokenizer,
            build_stamp=stamp,
        )
        role_items, role_stats = build_role_query(
            store,
            _seed_specs(config.seed_queries, "role_query"),
            config,
            tokenizer=tokenizer,
            build_stamp=stamp,
        )
    finally:
        store.close()

    items = sorted(
        [*content_items, *cross_items, *role_items], key=lambda item: item.query_id
    )
    stats = {
        "build": stamp,
        "content_grounded": content_stats,
        "cross_lingual": cross_stats,
        "role_query": role_stats,
        "totals": {
            "items": len(items),
            "tail_items": sum(1 for item in items if item.tail),
            "per_stratum": {
                stratum: sum(1 for item in items if item.stratum == stratum)
                for stratum in STRATA
            },
        },
    }
    return items, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SUBCOMMANDS = ("run", "snapshot", "build-goldset")


def _add_snapshot_source(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--snapshot", help="frozen snapshot produced by the `snapshot` command")
    group.add_argument(
        "--source-db",
        help="live database to freeze into a temporary snapshot first (opened read-only)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m living_memory.retrieval_harness",
        description="End-to-end retrieval harness over a frozen database snapshot.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser(
        "snapshot", help="freeze a database through the SQLite backup API"
    )
    snapshot.add_argument("--source-db", required=True, help="database to freeze (read-only)")
    snapshot.add_argument("--snapshot-out", required=True, help="destination snapshot path")

    run = subparsers.add_parser("run", help="replay a goldset end-to-end (default command)")
    _add_snapshot_source(run)
    run.add_argument("--goldset", required=True, help="goldset JSONL path")
    run.add_argument("--report", required=True, help="output JSON report path")
    run.add_argument("--markdown", help="output markdown report path")
    run.add_argument("--seed", type=int, default=0, help="seed for the agreement sample")
    run.add_argument("--cutoff", help="event cutoff recorded in provenance")
    run.add_argument(
        "--agreement-sample",
        type=int,
        default=DEFAULT_AGREEMENT_SAMPLE,
        help="items compared against an independently built live service (0 disables)",
    )
    run.add_argument(
        "--divergence-note",
        help="explanation that downgrades an agreement_rate below 1.0 from fatal to recorded",
    )
    run.add_argument(
        "--no-anchors",
        action="store_true",
        help="ablate the query-anchor entry into the graph channel "
        "(MemoryRecallService(anchor_seeding=False)). The anchor buckets are "
        "still annotated from the same snapshot, so this is the anchor-free arm "
        "of a with/without comparison and not a different measurement.",
    )

    goldset = subparsers.add_parser("build-goldset", help="generate a goldset from a snapshot")
    _add_snapshot_source(goldset)
    goldset.add_argument("--cutoff", required=True, help="ISO timestamp pinning the event slice")
    goldset.add_argument("--seed", type=int, default=0, help="sampling seed")
    goldset.add_argument("--seed-queries", help="curated cross_lingual / role_query spec JSON")
    goldset.add_argument("--out", required=True, help="output goldset JSONL path")
    goldset.add_argument("--build-report", help="optional JSON dump of the build statistics")
    goldset.add_argument("--max-content-grounded", type=int, help="cap for the content_grounded stratum")
    goldset.add_argument("--max-cross-lingual", type=int, help="cap for the cross_lingual stratum")
    goldset.add_argument("--max-role-query", type=int, help="cap for the role_query stratum")
    goldset.add_argument("--cross-lingual-top-k", type=int, default=5)
    goldset.add_argument(
        "--min-containment",
        type=float,
        default=LabelConfig().min_containment,
        help="replay containment threshold for the content_grounded label",
    )
    return parser


def _cmd_snapshot(args: argparse.Namespace) -> int:
    manifest = create_snapshot(args.source_db, args.snapshot_out)
    print(
        f"snapshot {manifest['snapshot_path']} sha256 {manifest['snapshot_sha256']} "
        f"({manifest['row_counts']['nodes']} nodes, "
        f"{manifest['row_counts']['recall_events']} recall_events) -> "
        f"{manifest_path_for(args.snapshot_out)}",
        file=sys.stderr,
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    goldset_path = Path(args.goldset)
    items = load_goldset(goldset_path)
    anchor_seeding = not args.no_anchors
    with frozen_snapshot(args.snapshot, args.source_db) as (snapshot_path, manifest):
        with working_copy(snapshot_path) as working_db:
            store = MemoryStore(MemoryConfig(db_path=working_db))
            try:
                service = MemoryRecallService(store, anchor_seeding=anchor_seeding)
                runs = run_goldset(service, items)
                # Annotated after the runs and independently of the ablation:
                # both arms see the same corpus, so both get the same buckets.
                annotations = annotate_anchors(service, items)
                anchor_corpus = {
                    "anchors": store.count_query_anchors(),
                    "live_anchors": store.count_query_anchors(include_decayed=False),
                    "edges": store.count_query_anchor_edges(),
                }
            finally:
                store.close()
            agreement = live_agreement(
                working_db,
                items,
                runs,
                sample_size=args.agreement_sample,
                seed=args.seed,
                anchor_seeding=anchor_seeding,
            )
        metrics = compute_metrics(runs, annotations)
        report = build_report(
            runs=runs,
            metrics=metrics,
            agreement=agreement,
            manifest=manifest,
            goldset_path=goldset_path,
            goldset_items=len(items),
            cutoff=normalize_cutoff(args.cutoff) if args.cutoff else None,
            seed=args.seed,
            divergence_note=args.divergence_note,
            annotations=annotations,
            anchor_seeding=anchor_seeding,
            anchor_corpus=anchor_corpus,
        )
    write_report(report, Path(args.report), Path(args.markdown) if args.markdown else None)

    overall = metrics["overall"]
    print(
        f"ran {metrics['items']} goldset items ({metrics['tail_items']} tail, "
        f"anchor_seeding={anchor_seeding}, "
        f"{anchor_corpus['live_anchors']} live anchors): "
        f"hit@1 {overall['hit@1']:.3f} hit@5 {overall['hit@5']:.3f} "
        f"mrr {overall['mrr']:.3f} -> {args.report}",
        file=sys.stderr,
    )
    rate = agreement["agreement_rate"]
    if rate is not None and rate < 1.0:
        message = (
            f"live-agreement rate {rate:.3f} < 1.0 over "
            f"{agreement['sampled_items']} sampled items "
            f"({len(agreement['divergences'])} divergences)"
        )
        if not args.divergence_note:
            print(f"FAIL: {message}; pass --divergence-note to record an explanation", file=sys.stderr)
            return 1
        print(f"WARN: {message}; note recorded: {args.divergence_note}", file=sys.stderr)
    return 0


def _cmd_build_goldset(args: argparse.Namespace) -> int:
    config = GoldsetBuildConfig(
        cutoff=args.cutoff,
        seed=args.seed,
        seed_queries=Path(args.seed_queries) if args.seed_queries else None,
        content_grounded_cap=args.max_content_grounded,
        cross_lingual_cap=args.max_cross_lingual,
        role_query_cap=args.max_role_query,
        cross_lingual_top_k=args.cross_lingual_top_k,
        min_containment=args.min_containment,
    )
    with frozen_snapshot(args.snapshot, args.source_db) as (snapshot_path, manifest):
        stamp = {
            "cutoff": normalize_cutoff(config.cutoff),
            "seed": config.seed,
            "harness_version": HARNESS_VERSION,
            "snapshot_sha256": manifest.get("snapshot_sha256"),
            "snapshot_captured_at": manifest.get("captured_at"),
            "source_db_path": manifest.get("source_path"),
            "git_commit": git_commit(),
            "embedding_backend": embedding_backend(),
        }
        with working_copy(snapshot_path) as working_db:
            items, stats = build_goldset(
                working_db, config, tokenizer=ModelVisibleSpan(), build_stamp=stamp
            )
    dump_goldset(items, args.out)
    # Re-read through the validator so a generated file can never be invalid.
    load_goldset(args.out)
    if args.build_report:
        report_path = Path(args.build_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "stats": stats,
                    "goldset_path": str(args.out),
                    "goldset_sha256": sha256_file(args.out),
                    "snapshot": dict(manifest),
                    "generated_at": utc_now_iso(),
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    totals = stats["totals"]
    print(
        f"built {totals['items']} goldset items "
        f"({totals['per_stratum']}, {totals['tail_items']} tail) -> {args.out}",
        file=sys.stderr,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        build_parser().print_help()
        return 2
    # Bare form: everything that is not an explicit subcommand runs `run`.
    if raw[0] not in SUBCOMMANDS and raw[0] not in ("-h", "--help"):
        raw = ["run", *raw]
    args = build_parser().parse_args(raw)
    if args.command == "snapshot":
        return _cmd_snapshot(args)
    if args.command == "build-goldset":
        return _cmd_build_goldset(args)
    return _cmd_run(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
