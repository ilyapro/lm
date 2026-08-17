#!/usr/bin/env python3
"""Mine two read-only handoff inventories from a Living Memory database.

This script is EVIDENCE-ONLY. It never writes to the database: the connection is
opened as ``file:<path>?mode=ro`` (uri=True) with ``PRAGMA query_only`` set, and
``MemoryStore`` is deliberately never imported (its constructor runs migrations
and therefore writes).

Two reports are produced under ``--out-dir`` (default ``artifacts/handoff``):

* ``schema-inventory.{json,md}`` -- every active ``level:schema`` node with its
  ``context.trigger``, provenance fields and firing statistics mined from
  ``recall_events.results``.
* ``oov-jargon.{json,md}`` -- a frequency dictionary of Cyrillic tokens seen in
  ``recall_events.query``, with corpus coverage, an out-of-vocabulary flag and a
  canonicalization *proposal* (never applied anywhere).

Determinism: every collection is sorted by an explicit key, floats are rounded,
and the generation timestamp doubles as the slice cutoff (``--as-of``). Re-running
with the recorded ``--as-of`` reproduces the same slice byte-for-byte, modulo the
``decayed`` flag which is always current state (documented in the reports).
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from typing import Any, Iterable

GENERATOR = "scripts/handoff_inventory.py"
GENERATOR_VERSION = "1.1.0"

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_SRC))

from living_memory.embeddings import (  # noqa: E402  (path bootstrap above)
    _RUSSIAN_STOP_WORDS as MODULE_RUSSIAN_STOP_WORDS,
    tokenize,
)
from living_memory.retrieval import (  # noqa: E402
    SCHEMA_TRIGGER_BASE_SCORE,
    SCHEMA_TRIGGER_OVERLAP_THRESHOLD,
)

DEFAULT_DB = Path.home() / ".local/share/living-memory/global.sqlite3"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "handoff"

# --------------------------------------------------------------------------
# Exact query slices (echoed into every report's provenance block).
# --------------------------------------------------------------------------
SQL_SCHEMAS = (
    "SELECT id, scope, context, content, created_at, timestamp, access_count, "
    "usefulness_score, confidence, unique_agents, last_accessed "
    "FROM nodes WHERE decayed = 0 AND level = 'schema' AND created_at <= :as_of "
    "ORDER BY id"
)
SQL_EVENTS = (
    "SELECT id, query, scope, resolved_scopes, session_id, transport_session_id, "
    "results, created_at, max_results "
    "FROM recall_events WHERE created_at <= :as_of ORDER BY id"
)
SQL_NODE_CONTENT = (
    "SELECT content, decayed FROM nodes WHERE created_at <= :as_of ORDER BY id"
)
SQL_FTS_DF_ACTIVE = (
    "SELECT count(*) FROM nodes_fts JOIN nodes n ON n.rowid = nodes_fts.rowid "
    "WHERE nodes_fts MATCH :token AND n.decayed = 0 AND n.created_at <= :as_of"
)

# --------------------------------------------------------------------------
# Determinism contract.
#
# `nodes` keeps only the LATEST value of these columns -- there is no history
# table to reconstruct "what was access_count at --as-of", so no cutoff can
# freeze them. Every other value in the analysis payload is derived from
# as-of-sliced rows and MUST be byte-stable across re-runs; `--verify` enforces
# exactly that split.
# --------------------------------------------------------------------------
VOLATILE_SCHEMA_FIELDS = (
    "access_count",
    "last_accessed",
    "usefulness_score",
    "confidence",
    "unique_agents",
)

# Provenance fields that describe the LIVE file / checkout rather than the
# analysis, and therefore move between runs by design.
VOLATILE_PROVENANCE_PATHS = frozenset(
    {
        "database.bytes",
        "database.mtime_utc",
        "git.commit",
        "git.branch",
        "git.worktree_dirty",
        "row_counts.nodes_total",
        "row_counts.nodes_active",
        "row_counts.recall_events_total",
        "row_counts.schema_nodes_active_in_slice",
        "row_counts.corpus_nodes_active",
        "runtime.python",
        "runtime.sqlite",
    }
)

DETERMINISM_CONTRACT = {
    "frozen_by_as_of": (
        "Every value in the analysis payload EXCEPT the fields listed in "
        "current_state_at_read. Re-running with the recorded --as-of against the "
        "same database must reproduce them byte-for-byte."
    ),
    "current_state_at_read": [f"schemas[].{field}" for field in VOLATILE_SCHEMA_FIELDS],
    "volatile_provenance_paths": sorted(VOLATILE_PROVENANCE_PATHS),
    "why_not_frozen": (
        "nodes stores only the latest value of these counters (no per-access "
        "history), so --as-of cannot reconstruct their value at the cutoff. They "
        "advance whenever the live MCP server serves a recall that touches the node."
    ),
    "why_decay_is_NOT_tolerated": (
        "corpus_df_active / fts_df_active / schema membership also depend on "
        "decayed = 0, which is current state -- but decay is a rare batch event, not "
        "continuous drift, so --verify deliberately reports it as frozen drift. That "
        "is the intended signal: the corpus moved under the artifact, regenerate it. "
        "The per-node counters above are excluded only because they advance on every "
        "served recall, which would make the check fire constantly and mean nothing."
    ),
    "generator_binding": (
        "provenance.git.generator_sha256 is checked as frozen, so --verify fails if "
        "the committed artifacts were produced by a different generator than the "
        "committed scripts/handoff_inventory.py."
    ),
    "verify_command": f"python3 {GENERATOR} --verify",
}

# --------------------------------------------------------------------------
# Cyrillic jargon extraction knobs.
# --------------------------------------------------------------------------
MIN_TOKEN_LENGTH = 3
CYRILLIC_TOKEN_RE = re.compile(r"^[а-я]+$")
LATIN_TOKEN_RE = re.compile(r"^[a-z][a-z0-9]*$")
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

# Function words that survive the shipped tokenizer's Russian stop list but
# carry no retrieval signal. Listed explicitly so the report can print it.
EXTRA_CYRILLIC_STOPWORDS = frozenset(
    {
        "быть",
        "был",
        "была",
        "были",
        "было",
        "вот",
        "все",
        "всех",
        "вы",
        "где",
        "да",
        "даже",
        "два",
        "две",
        "ему",
        "есть",
        "ещё",
        "еще",
        "здесь",
        "или",
        "их",
        "йот",
        "которая",
        "которые",
        "который",
        "кто",
        "куда",
        "мне",
        "мной",
        "может",
        "можно",
        "мы",
        "надо",
        "нам",
        "нас",
        "него",
        "нее",
        "нет",
        "них",
        "ним",
        "ничего",
        "ну",
        "нужно",
        "оно",
        "они",
        "очень",
        "потом",
        "потому",
        "почему",
        "просто",
        "раз",
        "себя",
        "сейчас",
        "тебя",
        "тем",
        "теперь",
        "тех",
        "того",
        "тоже",
        "той",
        "только",
        "том",
        "тот",
        "тут",
        "чем",
        "через",
        "чтобы",
        "эта",
        "этим",
        "этих",
        "этого",
        "этом",
        "этому",
        "являеться",
        "является",
    }
)

# Ranking: rank_score = occurrences * oov_factor, oov_factor = 1 / (1 + corpus_df_active).
# A token nobody stored (df 0) keeps its full frequency; a token stored in four
# nodes keeps a fifth of it. Documented in the MD report.
PROPOSAL_MAX_DF = 60  # only propose for tokens the corpus barely covers
PROPOSAL_MIN_OCCURRENCES = 2
BRIDGE_MIN_DF = 5  # a latin bridge target must be genuinely present
CYRILLIC_MIN_DF = 3
MAX_EDIT_DISTANCE = 2
MAX_TRANSLIT_VARIANTS = 24
PREFIX_MIN_LENGTH = 4
# A transliteration is only allowed to differ from its corpus target by one
# character below this length: without that gate every native Russian word finds
# some English lookalike (izuchi -> such) and the dictionary fills with noise.
LONG_WORD_LENGTH = 8
MIN_COMMON_PREFIX = 2
# Three-consonant skeletons (izuchi -> "skh" -> such) collide with unrelated
# English words; four is where the signal starts.
MIN_SKELETON_CONSONANTS = 4
MIN_SKELETON_SOURCE_LENGTH = 5

# Evidence strength of each proposal method. Only `strong` is quoted as a
# cross-language bridge in the Markdown; everything else is JSON-only material
# for the next goal to sift.
PROPOSAL_TIERS = {
    "curated": "strong",
    "latin-exact": "strong",
    "latin-skeleton": "medium",
    "cyrillic-stem": "medium",
    "latin-editdist": "weak",
    "cyrillic-editdist": "weak",
    "cyrillic-prefix": "weak",
    "in-corpus": "none",
    "none": "none",
    "not-evaluated": "none",
}

RUSSIAN_SUFFIXES = (
    "ами",
    "ями",
    "ов",
    "ев",
    "ой",
    "ей",
    "ий",
    "ый",
    "ые",
    "ая",
    "ое",
    "ом",
    "ем",
    "ах",
    "ях",
    "ам",
    "ям",
    "ую",
    "юю",
    "а",
    "е",
    "и",
    "о",
    "у",
    "ы",
    "ь",
    "й",
    "я",
    "ю",
)

TRANSLIT = {
    "а": ("a",),
    "б": ("b",),
    "в": ("v", "w"),
    "г": ("g",),
    "д": ("d",),
    "е": ("e",),
    "ж": ("zh", "j"),
    "з": ("z",),
    "и": ("i",),
    "й": ("y", "i"),
    "к": ("k", "c", "q"),
    "л": ("l",),
    "м": ("m",),
    "н": ("n",),
    "о": ("o",),
    "п": ("p",),
    "р": ("r",),
    "с": ("s",),
    "т": ("t",),
    "у": ("u", "oo"),
    "ф": ("f",),
    "х": ("h", "kh"),
    "ц": ("ts", "c"),
    "ч": ("ch",),
    "ш": ("sh",),
    "щ": ("sch",),
    "ъ": ("",),
    "ы": ("y", "i"),
    "ь": ("",),
    "э": ("e",),
    "ю": ("yu", "u"),
    "я": ("ya", "a"),
}
TRANSLIT_DIGRAPHS = {"дж": ("j", "g"), "кс": ("x", "ks")}

# Hand-authored canonicalization seeds. PROPOSALS ONLY -- this script does not
# touch tokenize()/_SYNONYMS, and neither does any artifact it writes.
CURATED_PROPOSALS: dict[str, str] = {
    "апрув": "approve",
    "апрувить": "approve",
    "апрувнуть": "approve",
    "апрувы": "approve",
    "аттач": "attachment",
    "аттача": "attachment",
    "мердж": "merge",
    "мерджа": "merge",
    "мерджить": "merge",
    "поревьювь": "review",
    "поревьюви": "review",
    "поревьювить": "review",
    "реквест": "request",
    "реквеста": "request",
    "реквесты": "request",
    "ревью": "review",
    "тултип": "tooltip",
    "тултипа": "tooltip",
    "тултипов": "tooltip",
    "тултипы": "tooltip",
    "туллтип": "tooltip",
    "туллтипа": "tooltip",
    "туллтипов": "tooltip",
    "туллтипы": "tooltip",
    "шаринга": "sharing",
}


# --------------------------------------------------------------------------
# Read-only database access.
# --------------------------------------------------------------------------
def open_readonly(path: Path) -> sqlite3.Connection:
    """Open ``path`` strictly read-only and pin one consistent WAL snapshot."""

    if not path.exists():
        raise SystemExit(f"database not found: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = 1")
    query_only = conn.execute("PRAGMA query_only").fetchone()[0]
    if not query_only:  # pragma: no cover - defensive
        raise SystemExit("refusing to continue: connection is not query_only")
    # A deferred read transaction pins one WAL snapshot for the whole run so the
    # counts in the two reports describe the same instant of a live database.
    conn.execute("BEGIN")
    conn.execute("SELECT count(*) FROM metadata").fetchone()
    return conn


def git_provenance() -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                args,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):  # pragma: no cover
            return None
        if out.returncode != 0:
            return None
        return out.stdout.strip()

    head = run(["git", "rev-parse", "HEAD"])
    status = run(["git", "status", "--porcelain"])
    generator_path = REPO_ROOT / GENERATOR
    try:
        generator_sha256 = hashlib.sha256(generator_path.read_bytes()).hexdigest()
    except OSError:  # pragma: no cover - generator always readable in practice
        generator_sha256 = None
    return {
        "commit": head,
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "worktree_dirty": bool(status) if status is not None else None,
        "generator_sha256": generator_sha256,
        "note": (
            "`commit` is HEAD at generation time and is REBASE-FRAGILE: the artifacts "
            "land in a later commit, and any rebase of this branch rewrites both SHAs, "
            "so a recorded commit can stop being reachable from HEAD. The durable "
            "anchor is `generator_sha256` -- the content hash of the generator that "
            "produced these bytes, which survives rebase. Verify with: "
            "`sha256sum scripts/handoff_inventory.py`."
        ),
    }


# --------------------------------------------------------------------------
# Tokenization helpers.
# --------------------------------------------------------------------------
def surface_tokens(text: str) -> list[str]:
    """Surface tokens under the shipped normalization, minus stemming/synonyms.

    Mirrors ``living_memory.embeddings.tokenize`` up to (but excluding)
    ``_canonical_token``: camelCase split, ``_-/.`` -> space, lowercase, yo-fold,
    ``\\w+``. Cyrillic tokens are unaffected by ``_stem`` (latin-only) and by
    ``_SYNONYMS`` (latin-only keys), so for the Cyrillic dictionary the surface
    form IS the form the bm25 channel searches for.
    """

    separated = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    separated = separated.replace("_", " ").replace("-", " ").replace("/", " ").replace(".", " ")
    normalized = separated.lower().replace("ё", "е")
    return _TOKEN_RE.findall(normalized)


def build_stoplist() -> tuple[frozenset[str], list[str], list[str]]:
    module_words = sorted(w for w in MODULE_RUSSIAN_STOP_WORDS if len(w) >= MIN_TOKEN_LENGTH)
    extra_words = sorted(EXTRA_CYRILLIC_STOPWORDS)
    return frozenset(module_words) | EXTRA_CYRILLIC_STOPWORDS, module_words, extra_words


def bounded_levenshtein(left: str, right: str, max_distance: int) -> int:
    """Levenshtein distance, giving up (``max_distance + 1``) once exceeded."""

    if abs(len(left) - len(right)) > max_distance:
        return max_distance + 1
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for i, lc in enumerate(left, start=1):
        current = [i]
        row_best = i
        for j, rc in enumerate(right, start=1):
            cost = 0 if lc == rc else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            row_best = min(row_best, value)
        if row_best > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def russian_stems(token: str) -> list[str]:
    """Token plus its suffix-stripped stems, longest first, deterministic."""

    stems = {token}
    for suffix in RUSSIAN_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            stems.add(token[: -len(suffix)])
    return sorted(stems, key=lambda s: (-len(s), s))


def translit_variants(token: str) -> list[str]:
    """Bounded, deterministic set of latin transliterations of a Cyrillic token."""

    variants = [""]
    index = 0
    while index < len(token):
        digraph = token[index : index + 2]
        if digraph in TRANSLIT_DIGRAPHS:
            options = TRANSLIT_DIGRAPHS[digraph] + (
                "".join(TRANSLIT.get(ch, (ch,))[0] for ch in digraph),
            )
            index += 2
        else:
            options = TRANSLIT.get(token[index], (token[index],))
            index += 1
        grown = [prefix + option for prefix in variants for option in options]
        # Keep the search bounded but stable: dedupe, sort, truncate.
        variants = sorted(dict.fromkeys(grown))[:MAX_TRANSLIT_VARIANTS]
    return [v for v in variants if v]


_VOWELS = frozenset("aeiouy")
_SKELETON_MAP = {"j": "g", "c": "k", "q": "k", "w": "v", "z": "s"}


def skeleton(word: str) -> str:
    """Consonant skeleton: phonetic folding, vowels dropped, doubles collapsed.

    ``tooltip`` and ``tulltip`` both become ``tltp``; ``merge`` and ``merj``
    both become ``mrg``; ``request`` and ``rekvest`` both become ``rkvst``.
    Lossy by construction -- every skeleton match is reported as such so the
    next goal can weigh the evidence.
    """

    folded = word.lower().replace("ph", "f").replace("ck", "k")
    out: list[str] = []
    for char in folded:
        char = _SKELETON_MAP.get(char, char)
        if char in _VOWELS:
            continue
        if out and out[-1] == char:
            continue
        out.append(char)
    return "".join(out)


class VocabularyIndex:
    """Corpus vocabulary with the lookups the proposal engine needs."""

    def __init__(self, df_active: Counter[str], min_df: int) -> None:
        self.df_active = df_active
        self.min_df = min_df
        self.words = sorted(word for word, df in df_active.items() if df >= min_df)
        self.word_set = set(self.words)
        self.deletes: dict[str, list[str]] = defaultdict(list)
        self.skeletons: dict[str, list[str]] = defaultdict(list)
        for word in self.words:
            for variant in self._delete_variants(word):
                self.deletes[variant].append(word)
            self.skeletons[skeleton(word)].append(word)

    @staticmethod
    def _delete_variants(word: str) -> set[str]:
        """All strings obtained by deleting up to ``MAX_EDIT_DISTANCE`` chars."""

        current = {word}
        seen = {word}
        for _ in range(MAX_EDIT_DISTANCE):
            grown = {w[:i] + w[i + 1 :] for w in current for i in range(len(w))}
            grown -= seen
            seen |= grown
            current = grown
        return seen

    def fuzzy(self, word: str) -> list[tuple[int, int, str]]:
        """Candidates within ``MAX_EDIT_DISTANCE``: (distance, -df, word)."""

        candidates: set[str] = set()
        for variant in self._delete_variants(word):
            candidates.update(self.deletes.get(variant, ()))
        scored = []
        for candidate in sorted(candidates):
            distance = bounded_levenshtein(word, candidate, MAX_EDIT_DISTANCE)
            if distance <= MAX_EDIT_DISTANCE:
                scored.append((distance, -self.df_active[candidate], candidate))
        return sorted(scored)

    def by_skeleton(self, word: str, max_length_delta: int = 2) -> list[tuple[int, str]]:
        """Candidates sharing the consonant skeleton: (-df, word)."""

        key = skeleton(word)
        if len(key) < MIN_SKELETON_CONSONANTS:
            return []
        hits = [
            (-self.df_active[candidate], candidate)
            for candidate in self.skeletons.get(key, ())
            if abs(len(candidate) - len(word)) <= max_length_delta
        ]
        return sorted(hits)

    def prefix_completions(self, prefix: str) -> list[tuple[int, str]]:
        """In-corpus words that ``prefix`` is a strict prefix of: (-df, word)."""

        start = bisect_left(self.words, prefix)
        hits = []
        for word in self.words[start : start + 200]:
            if not word.startswith(prefix):
                break
            if word != prefix:
                hits.append((-self.df_active[word], word))
        return sorted(hits)


def _common_prefix_length(left: str, right: str) -> int:
    length = 0
    for lc, rc in zip(left, right):
        if lc != rc:
            break
        length += 1
    return length


def _plausible_bridge(variant: str, hit: tuple[int, int, str]) -> bool:
    """Reject transliteration matches that are lookalikes rather than borrowings."""

    distance, _negative_df, word = hit
    allowed = 1 if len(variant) < LONG_WORD_LENGTH else MAX_EDIT_DISTANCE
    if distance > allowed:
        return False
    return _common_prefix_length(variant, word) >= MIN_COMMON_PREFIX


def propose_canonicalization(
    token: str,
    *,
    cyrillic_vocab: VocabularyIndex,
    latin_vocab: VocabularyIndex,
) -> dict[str, Any]:
    """Return a PROPOSAL for ``token`` -- never an edit, never applied.

    Method priority, strongest evidence first:
      ``curated`` > ``latin-exact`` > ``latin-editdist`` > ``latin-skeleton``
      > ``cyrillic-stem`` > ``cyrillic-editdist`` > ``cyrillic-prefix``.
    """

    empty = {
        "status": "PROPOSAL",
        "canonical": None,
        "method": "none",
        "evidence": {},
        "alternatives": [],
    }

    curated = CURATED_PROPOSALS.get(token)
    if curated is not None:
        return {
            "status": "PROPOSAL",
            "canonical": curated,
            "method": "curated",
            "evidence": {
                "source": "hand-authored CURATED_PROPOSALS table in " + GENERATOR,
                "target_corpus_df": latin_vocab.df_active.get(curated, 0),
            },
            "alternatives": [],
        }

    stems = russian_stems(token)

    # --- latin bridge: the cross-language class the root goal cares about ---
    for stem in stems:
        for variant in translit_variants(stem):
            if variant in latin_vocab.word_set and latin_vocab.df_active[variant] >= BRIDGE_MIN_DF:
                return {
                    "status": "PROPOSAL",
                    "canonical": variant,
                    "method": "latin-exact",
                    "evidence": {
                        "stem": stem,
                        "translit": variant,
                        "target_corpus_df": latin_vocab.df_active[variant],
                    },
                    "alternatives": [],
                }
    for stem in stems:
        for variant in translit_variants(stem):
            if len(variant) < MIN_SKELETON_SOURCE_LENGTH:
                continue
            hits = latin_vocab.by_skeleton(variant)
            if hits:
                negative_df, word = hits[0]
                return {
                    "status": "PROPOSAL",
                    "canonical": word,
                    "method": "latin-skeleton",
                    "evidence": {
                        "stem": stem,
                        "translit": variant,
                        "skeleton": skeleton(variant),
                        "target_corpus_df": -negative_df,
                    },
                    "alternatives": [w for _, w in hits[1:4]],
                }
    # --- russian side: inflection, typo, truncated query ---
    if token in cyrillic_vocab.word_set:
        # The corpus already stores this exact word: bm25 can reach it, so there
        # is nothing to canonicalize on the Cyrillic side.
        return {
            "status": "PROPOSAL",
            "canonical": None,
            "method": "in-corpus",
            "evidence": {"corpus_df_active": cyrillic_vocab.df_active[token]},
            "alternatives": [],
        }
    for stem in stems[1:]:
        if stem in cyrillic_vocab.word_set:
            return {
                "status": "PROPOSAL",
                "canonical": stem,
                "method": "cyrillic-stem",
                "evidence": {"target_corpus_df": cyrillic_vocab.df_active[stem]},
                "alternatives": [],
            }
    hits = [hit for hit in cyrillic_vocab.fuzzy(token) if hit[2] != token]
    if hits:
        distance, negative_df, word = hits[0]
        return {
            "status": "PROPOSAL",
            "canonical": word,
            "method": "cyrillic-editdist",
            "evidence": {"edit_distance": distance, "target_corpus_df": -negative_df},
            "alternatives": [w for _, _, w in hits[1:4]],
        }
    if len(token) >= PREFIX_MIN_LENGTH:
        completions = cyrillic_vocab.prefix_completions(token)
        if completions:
            negative_df, word = completions[0]
            return {
                "status": "PROPOSAL",
                "canonical": word,
                "method": "cyrillic-prefix",
                "evidence": {
                    "target_corpus_df": -negative_df,
                    "reading": "query token looks truncated mid-word",
                },
                "alternatives": [w for _, w in completions[1:4]],
            }

    # --- last resort: a tight transliteration near-miss ---
    for stem in stems:
        for variant in translit_variants(stem):
            hits = [hit for hit in latin_vocab.fuzzy(variant) if _plausible_bridge(variant, hit)]
            if hits:
                distance, negative_df, word = hits[0]
                return {
                    "status": "PROPOSAL",
                    "canonical": word,
                    "method": "latin-editdist",
                    "evidence": {
                        "stem": stem,
                        "translit": variant,
                        "edit_distance": distance,
                        "target_corpus_df": -negative_df,
                    },
                    "alternatives": [w for _, _, w in hits[1:4]],
                }
    return empty


# --------------------------------------------------------------------------
# Mining.
# --------------------------------------------------------------------------
def load_schemas(conn: sqlite3.Connection, as_of: str) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    for row in conn.execute(SQL_SCHEMAS, {"as_of": as_of}):
        try:
            context = json.loads(row["context"] or "{}")
        except json.JSONDecodeError:
            context = {}
        if not isinstance(context, dict):
            context = {}
        trigger = str(context.get("trigger") or "")
        trigger_tokens = sorted(set(tokenize(trigger)))
        schemas.append(
            {
                "id": row["id"],
                "scope": row["scope"],
                "task_pattern": context.get("task_pattern"),
                "procedure_id": context.get("procedure_id"),
                "trigger": trigger,
                "trigger_tokens": trigger_tokens,
                "trigger_token_count": len(trigger_tokens),
                "content_chars": len(row["content"] or ""),
                "created_at": row["created_at"],
                "timestamp": row["timestamp"],
                "access_count": row["access_count"],
                "usefulness_score": round(float(row["usefulness_score"] or 0.0), 6),
                "confidence": round(float(row["confidence"] or 0.0), 6),
                "unique_agents": row["unique_agents"],
                "last_accessed": row["last_accessed"],
            }
        )
    return schemas


class EventScan:
    """One pass over ``recall_events`` feeding both reports."""

    def __init__(self, schemas: list[dict[str, Any]], stoplist: frozenset[str]) -> None:
        self.stoplist = stoplist
        self.schema_index = {schema["id"]: schema for schema in schemas}
        self.trigger_sets = {
            schema["id"]: frozenset(schema["trigger_tokens"])
            for schema in schemas
            if schema["trigger_tokens"]
        }
        self.inverted: dict[str, list[str]] = defaultdict(list)
        for schema_id, tokens in self.trigger_sets.items():
            for token in tokens:
                self.inverted[token].append(schema_id)
        self.needed = {
            schema_id: SCHEMA_TRIGGER_OVERLAP_THRESHOLD * len(tokens)
            for schema_id, tokens in self.trigger_sets.items()
        }

        # schema-side accumulators
        self.delivered = Counter()
        self.trigger_hits = Counter()
        self.other_channel_hits = Counter()
        self.channel_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.best_rank: dict[str, int] = {}
        self.rank1 = Counter()
        self.top3 = Counter()
        self.first_delivered: dict[str, str] = {}
        self.last_delivered: dict[str, str] = {}
        self.delivered_sessions: dict[str, set[str]] = defaultdict(set)
        self.delivered_scopes: dict[str, set[str]] = defaultdict(set)
        self.delivered_queries: dict[str, set[str]] = defaultdict(set)
        self.eligible_events = Counter()
        self.simulated_matches = Counter()
        self.crowded_matches = Counter()
        self.trigger_hits_per_event = Counter()
        self.simulated_matches_per_event = Counter()

        # jargon-side accumulators
        self.token_occurrences = Counter()
        self.token_events = Counter()
        self.token_sessions: dict[str, set[str]] = defaultdict(set)
        self.token_scopes: dict[str, Counter[str]] = defaultdict(Counter)
        self.token_queries: dict[str, set[str]] = defaultdict(set)
        self.stoplisted_tokens = Counter()

        self.event_count = 0
        self.result_rows = 0
        self.events_without_tokens = 0
        self.events_without_session = 0
        self.first_event_at: str | None = None
        self.last_event_at: str | None = None

    def run(self, conn: sqlite3.Connection, as_of: str) -> None:
        schema_created = {sid: self.schema_index[sid]["created_at"] for sid in self.schema_index}
        for row in conn.execute(SQL_EVENTS, {"as_of": as_of}):
            self.event_count += 1
            created_at = row["created_at"]
            if self.first_event_at is None:
                self.first_event_at = created_at
            self.last_event_at = created_at
            query = row["query"] or ""
            session_key = row["session_id"] or row["transport_session_id"] or ""
            if not session_key:
                self.events_without_session += 1
            scope = row["scope"] or "global"

            self._scan_results(row, created_at, session_key, scope, query)
            self._scan_simulation(row, created_at, scope, schema_created)
            self._scan_jargon(query, session_key, scope)

    # -- delivered results ------------------------------------------------
    def _scan_results(
        self,
        row: sqlite3.Row,
        created_at: str,
        session_key: str,
        scope: str,
        query: str,
    ) -> None:
        try:
            results = json.loads(row["results"] or "[]")
        except json.JSONDecodeError:
            return
        if not isinstance(results, list):
            return
        triggered_here = 0
        for item in results:
            if not isinstance(item, dict):
                continue
            self.result_rows += 1
            node_id = item.get("node_id")
            if node_id not in self.schema_index:
                continue
            self.delivered[node_id] += 1
            trigger_score = float(item.get("trigger_score") or 0.0)
            if trigger_score > 0.0:
                self.trigger_hits[node_id] += 1
                triggered_here += 1
            else:
                self.other_channel_hits[node_id] += 1
            for method in item.get("methods") or []:
                self.channel_counts[node_id][str(method)] += 1
            rank = item.get("rank")
            if isinstance(rank, int):
                previous = self.best_rank.get(node_id)
                self.best_rank[node_id] = rank if previous is None else min(previous, rank)
                if rank == 1:
                    self.rank1[node_id] += 1
                if rank <= 3:
                    self.top3[node_id] += 1
            if node_id not in self.first_delivered:
                self.first_delivered[node_id] = created_at
            self.last_delivered[node_id] = created_at
            if session_key:
                self.delivered_sessions[node_id].add(session_key)
            self.delivered_scopes[node_id].add(scope)
            self.delivered_queries[node_id].add(query)
        if triggered_here:
            self.trigger_hits_per_event[triggered_here] += 1

    # -- simulated trigger channel ---------------------------------------
    def _scan_simulation(
        self,
        row: sqlite3.Row,
        created_at: str,
        scope: str,
        schema_created: dict[str, str],
    ) -> None:
        try:
            resolved = json.loads(row["resolved_scopes"] or "[]")
        except json.JSONDecodeError:
            resolved = []
        scopes = set(resolved) if isinstance(resolved, list) else set()
        scopes.add(scope)

        eligible = [
            schema_id
            for schema_id, created in schema_created.items()
            if created <= created_at and self.schema_index[schema_id]["scope"] in scopes
        ]
        for schema_id in eligible:
            self.eligible_events[schema_id] += 1
        if not eligible:
            return
        eligible_set = set(eligible)

        query_tokens = set(tokenize(row["query"] or ""))
        if not query_tokens:
            self.events_without_tokens += 1
            return
        overlaps = Counter()
        for token in query_tokens:
            for schema_id in self.inverted.get(token, ()):
                overlaps[schema_id] += 1
        max_results = int(row["max_results"] or 0)
        matching = sorted(
            schema_id
            for schema_id, hits in overlaps.items()
            if schema_id in eligible_set and hits >= self.needed[schema_id]
        )
        if not matching:
            return
        self.simulated_matches_per_event[len(matching)] += 1
        crowded = bool(max_results) and len(matching) > max_results
        for schema_id in matching:
            self.simulated_matches[schema_id] += 1
            if crowded:
                # More schemas fired the trigger than the recall could return:
                # this match competed for a slot it could not all get.
                self.crowded_matches[schema_id] += 1

    # -- cyrillic jargon --------------------------------------------------
    def _scan_jargon(self, query: str, session_key: str, scope: str) -> None:
        seen: set[str] = set()
        for token in surface_tokens(query):
            if len(token) < MIN_TOKEN_LENGTH:
                continue
            if not CYRILLIC_TOKEN_RE.match(token):
                continue
            if token in self.stoplist:
                self.stoplisted_tokens[token] += 1
                continue
            self.token_occurrences[token] += 1
            if session_key:
                self.token_sessions[token].add(session_key)
            self.token_scopes[token][scope] += 1
            self.token_queries[token].add(query)
            seen.add(token)
        for token in seen:
            self.token_events[token] += 1


def build_corpus_df(conn: sqlite3.Connection, as_of: str) -> tuple[Counter[str], Counter[str], int, int]:
    """Document frequency of every surface token over ``nodes.content``."""

    df_active: Counter[str] = Counter()
    df_all: Counter[str] = Counter()
    active_nodes = 0
    all_nodes = 0
    for row in conn.execute(SQL_NODE_CONTENT, {"as_of": as_of}):
        all_nodes += 1
        tokens = set(surface_tokens(row["content"] or ""))
        df_all.update(tokens)
        if not row["decayed"]:
            active_nodes += 1
            df_active.update(tokens)
    return df_active, df_all, active_nodes, all_nodes


def fts_document_frequency(conn: sqlite3.Connection, token: str, as_of: str) -> int:
    """Active-node hit count for ``token`` on the real bm25 index path.

    Sliced by ``as_of`` for the same reason ``SQL_NODE_CONTENT`` is: without the
    cutoff this counts nodes written after the slice, so ``fts_df_active`` drifts
    upward on every re-run and is measured over a different node set than
    ``corpus_df_active`` -- which would make the two DF columns incomparable.
    """

    escaped = '"' + token.replace('"', '""') + '"'
    try:
        row = conn.execute(SQL_FTS_DF_ACTIVE, {"token": escaped, "as_of": as_of}).fetchone()
        return int(row[0])
    except sqlite3.OperationalError:  # pragma: no cover - malformed MATCH
        return -1


# --------------------------------------------------------------------------
# Trigger collisions.
# --------------------------------------------------------------------------
def find_collisions(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    """Pairs of schemas whose SHARED trigger tokens alone fire both.

    The live rule (retrieval.py:460-465) fires a schema when
    ``|query_tokens & trigger_tokens| / |trigger_tokens| >= 0.5``. So if
    ``|A & B| >= 0.5*|A|`` and ``|A & B| >= 0.5*|B|``, the query consisting of
    nothing but the shared tokens fires BOTH -- a guaranteed collision, not a
    similarity heuristic.
    """

    entries = [
        (schema["id"], schema["scope"], frozenset(schema["trigger_tokens"]))
        for schema in schemas
        if schema["trigger_tokens"]
    ]
    pairs: list[dict[str, Any]] = []
    parent: dict[str, str] = {schema_id: schema_id for schema_id, _, _ in entries}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    identical_groups: dict[frozenset[str], list[str]] = defaultdict(list)
    for schema_id, _scope, tokens in entries:
        identical_groups[tokens].append(schema_id)

    for i, (left_id, left_scope, left_tokens) in enumerate(entries):
        for right_id, right_scope, right_tokens in entries[i + 1 :]:
            shared = left_tokens & right_tokens
            if not shared:
                continue
            if (
                len(shared) >= SCHEMA_TRIGGER_OVERLAP_THRESHOLD * len(left_tokens)
                and len(shared) >= SCHEMA_TRIGGER_OVERLAP_THRESHOLD * len(right_tokens)
            ):
                union(left_id, right_id)
                pairs.append(
                    {
                        "a": left_id,
                        "b": right_id,
                        "a_scope": left_scope,
                        "b_scope": right_scope,
                        "same_scope": left_scope == right_scope,
                        "shared_tokens": sorted(shared),
                        "identical_token_sets": left_tokens == right_tokens,
                    }
                )

    clusters: dict[str, list[str]] = defaultdict(list)
    for schema_id, _scope, _tokens in entries:
        clusters[find(schema_id)].append(schema_id)
    sized = sorted(
        (
            {"root": root, "size": len(members), "members": sorted(members)}
            for root, members in clusters.items()
            if len(members) > 1
        ),
        key=lambda cluster: (-cluster["size"], cluster["root"]),
    )
    identical = sorted(
        (
            {
                "tokens": sorted(tokens),
                "count": len(members),
                "node_ids": sorted(members),
            }
            for tokens, members in identical_groups.items()
            if len(members) > 1
        ),
        key=lambda group: (-group["count"], group["tokens"]),
    )
    return {
        "rule": (
            "shared >= 0.5*|A| and shared >= 0.5*|B| -- the shared tokens alone "
            "fire both schemas under retrieval.py:461"
        ),
        "pair_count": len(pairs),
        "colliding_schema_count": sum(cluster["size"] for cluster in sized),
        "cluster_count": len(sized),
        "clusters": sized,
        "identical_trigger_token_sets": identical,
        "pairs": sorted(pairs, key=lambda pair: (pair["a"], pair["b"])),
    }


# --------------------------------------------------------------------------
# Report assembly.
# --------------------------------------------------------------------------
MAX_COLLISION_PAIRS_IN_JSON = 500


def build_schema_report(
    schemas: list[dict[str, Any]],
    scan: EventScan,
    collisions: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    for schema in schemas:
        schema_id = schema["id"]
        delivered = scan.delivered[schema_id]
        trigger_hits = scan.trigger_hits[schema_id]
        simulated = scan.simulated_matches[schema_id]
        record = dict(schema)
        record["firing"] = {
            "delivered_events": delivered,
            "trigger_channel_hits": trigger_hits,
            "other_channel_hits": scan.other_channel_hits[schema_id],
            "channel_counts": dict(sorted(scan.channel_counts[schema_id].items())),
            "best_rank": scan.best_rank.get(schema_id),
            "rank1_hits": scan.rank1[schema_id],
            "top3_hits": scan.top3[schema_id],
            "first_delivered_at": scan.first_delivered.get(schema_id),
            "last_delivered_at": scan.last_delivered.get(schema_id),
            "distinct_sessions": len(scan.delivered_sessions[schema_id]),
            "distinct_scopes": len(scan.delivered_scopes[schema_id]),
            "distinct_queries": len(scan.delivered_queries[schema_id]),
            "eligible_events": scan.eligible_events[schema_id],
            "simulated_trigger_matches": simulated,
            "crowded_trigger_matches": scan.crowded_matches[schema_id],
            "delivered_share_of_matches": (
                round(trigger_hits / simulated, 6) if simulated else None
            ),
        }
        nodes.append(record)

    never_delivered = [n["id"] for n in nodes if n["firing"]["delivered_events"] == 0]
    never_triggered = [n["id"] for n in nodes if n["firing"]["trigger_channel_hits"] == 0]
    other_only = [
        n["id"]
        for n in nodes
        if n["firing"]["delivered_events"] > 0 and n["firing"]["trigger_channel_hits"] == 0
    ]
    never_matched = [n["id"] for n in nodes if n["firing"]["simulated_trigger_matches"] == 0]
    matched_never_delivered = [
        n["id"]
        for n in nodes
        if n["firing"]["simulated_trigger_matches"] > 0 and n["firing"]["delivered_events"] == 0
    ]

    top_firers = sorted(
        nodes,
        key=lambda n: (-n["firing"]["delivered_events"], n["id"]),
    )[:60]
    trigger_texts = Counter(n["trigger"] for n in nodes)
    duplicate_triggers = sorted(
        (
            {
                "trigger": trigger,
                "count": count,
                "node_ids": sorted(n["id"] for n in nodes if n["trigger"] == trigger),
                "scopes": dict(
                    sorted(Counter(n["scope"] for n in nodes if n["trigger"] == trigger).items())
                ),
            }
            for trigger, count in trigger_texts.items()
            if count > 1
        ),
        key=lambda entry: (-entry["count"], entry["trigger"]),
    )
    hair_triggers = sorted(
        (
            {
                "id": n["id"],
                "scope": n["scope"],
                "trigger": n["trigger"],
                "trigger_tokens": n["trigger_tokens"],
                "simulated_trigger_matches": n["firing"]["simulated_trigger_matches"],
                "trigger_channel_hits": n["firing"]["trigger_channel_hits"],
            }
            for n in nodes
            if 0 < n["trigger_token_count"] <= 2
        ),
        key=lambda entry: (-entry["simulated_trigger_matches"], entry["id"]),
    )

    collisions_json = dict(collisions)
    all_pairs = collisions_json.pop("pairs")
    collisions_json["pairs_included"] = min(len(all_pairs), MAX_COLLISION_PAIRS_IN_JSON)
    collisions_json["pairs_truncated"] = len(all_pairs) > MAX_COLLISION_PAIRS_IN_JSON
    collisions_json["pairs"] = all_pairs[:MAX_COLLISION_PAIRS_IN_JSON]

    summary = {
        "active_schema_count": len(nodes),
        "with_trigger": sum(1 for n in nodes if n["trigger"]),
        "with_trigger_tokens": sum(1 for n in nodes if n["trigger_tokens"]),
        "with_task_pattern": sum(1 for n in nodes if n["task_pattern"]),
        "with_procedure_id": sum(1 for n in nodes if n["procedure_id"]),
        "scopes": dict(sorted(Counter(n["scope"] for n in nodes).items())),
        "distinct_trigger_texts": len(trigger_texts),
        "distinct_trigger_token_sets": len({tuple(n["trigger_tokens"]) for n in nodes}),
        "trigger_token_count_histogram": dict(
            sorted(Counter(n["trigger_token_count"] for n in nodes).items())
        ),
        "content_chars": {
            "min": min((n["content_chars"] for n in nodes), default=0),
            "median": _median([n["content_chars"] for n in nodes]),
            "max": max((n["content_chars"] for n in nodes), default=0),
            "total": sum(n["content_chars"] for n in nodes),
        },
        "delivery": {
            "total_schema_deliveries": sum(scan.delivered.values()),
            "total_trigger_channel_hits": sum(scan.trigger_hits.values()),
            "total_other_channel_hits": sum(scan.other_channel_hits.values()),
            "delivered_schema_count": len(nodes) - len(never_delivered),
            "never_delivered_count": len(never_delivered),
            "never_trigger_fired_count": len(never_triggered),
            "delivered_only_via_other_channels_count": len(other_only),
            "channel_counts": dict(sorted(_merge_counters(scan.channel_counts.values()).items())),
            "trigger_hits_per_event_histogram": dict(sorted(scan.trigger_hits_per_event.items())),
        },
        "simulation": {
            "what_it_is": (
                "replay of the live trigger rule (retrieval.py:443-465) over every "
                "recorded query: scope-filtered by recall_events.resolved_scopes and "
                "restricted to events at or after the schema's created_at"
            ),
            "total_simulated_matches": sum(scan.simulated_matches.values()),
            "matched_schema_count": len(nodes) - len(never_matched),
            "never_matched_count": len(never_matched),
            "matched_but_never_delivered_count": len(matched_never_delivered),
            "total_crowded_matches": sum(scan.crowded_matches.values()),
            "matches_per_event_histogram": dict(sorted(scan.simulated_matches_per_event.items())),
        },
        "never_delivered_ids": sorted(never_delivered),
        "matched_but_never_delivered_ids": sorted(matched_never_delivered),
        "never_matched_ids": sorted(never_matched),
        "top_firers": [
            {
                "id": n["id"],
                "scope": n["scope"],
                "trigger": n["trigger"],
                "delivered_events": n["firing"]["delivered_events"],
                "trigger_channel_hits": n["firing"]["trigger_channel_hits"],
                "simulated_trigger_matches": n["firing"]["simulated_trigger_matches"],
                "access_count": n["access_count"],
            }
            for n in top_firers
        ],
        "duplicate_trigger_texts": duplicate_triggers,
        "hair_triggers": hair_triggers,
        "collisions": collisions_json,
    }
    return {
        "report": "schema-inventory",
        "provenance": provenance,
        "trigger_rule": {
            "source": "src/living_memory/retrieval.py:437-465",
            "overlap_threshold": SCHEMA_TRIGGER_OVERLAP_THRESHOLD,
            "base_score": SCHEMA_TRIGGER_BASE_SCORE,
            "formula": "overlap = |tokenize(query) & tokenize(trigger)| / |tokenize(trigger)|",
        },
        "summary": summary,
        "schemas": sorted(nodes, key=lambda n: n["id"]),
    }


def _merge_counters(counters: Iterable[Counter[str]]) -> Counter[str]:
    merged: Counter[str] = Counter()
    for counter in counters:
        merged.update(counter)
    return merged


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return round((ordered[middle - 1] + ordered[middle]) / 2, 6)


def _df_band(df: int) -> str:
    if df == 0:
        return "0"
    if df == 1:
        return "1"
    if df <= 5:
        return "2-5"
    if df <= 20:
        return "6-20"
    if df <= 100:
        return "21-100"
    return ">100"


def build_jargon_report(
    conn: sqlite3.Connection,
    scan: EventScan,
    df_active: Counter[str],
    df_all: Counter[str],
    module_stopwords: list[str],
    extra_stopwords: list[str],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    cyrillic_vocab = VocabularyIndex(
        Counter({w: c for w, c in df_active.items() if CYRILLIC_TOKEN_RE.match(w)}),
        CYRILLIC_MIN_DF,
    )
    latin_vocab = VocabularyIndex(
        Counter({w: c for w, c in df_active.items() if LATIN_TOKEN_RE.match(w)}),
        BRIDGE_MIN_DF,
    )

    as_of = provenance["as_of"]
    tokens: list[dict[str, Any]] = []
    for token, occurrences in scan.token_occurrences.items():
        active_df = df_active.get(token, 0)
        all_df = df_all.get(token, 0)
        fts_df = fts_document_frequency(conn, token, as_of)
        is_oov = active_df == 0 and fts_df <= 0
        oov_factor = 1.0 / (1.0 + active_df)
        scopes = scan.token_scopes[token]
        record = {
            "token": token,
            "occurrences": occurrences,
            "distinct_queries": len(scan.token_queries[token]),
            "distinct_events": scan.token_events[token],
            "distinct_sessions": len(scan.token_sessions[token]),
            "distinct_scopes": len(scopes),
            "top_scopes": [
                {"scope": scope, "count": count}
                for scope, count in sorted(scopes.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            ],
            "corpus_df_active": active_df,
            "corpus_df_all_nodes": all_df,
            "fts_df_active": fts_df,
            "df_band": _df_band(active_df),
            "is_oov": is_oov,
            "oov_factor": round(oov_factor, 6),
            "rank_score": round(occurrences * oov_factor, 6),
        }
        if occurrences >= PROPOSAL_MIN_OCCURRENCES and active_df <= PROPOSAL_MAX_DF:
            proposal = propose_canonicalization(
                token, cyrillic_vocab=cyrillic_vocab, latin_vocab=latin_vocab
            )
        else:
            proposal = {
                "status": "PROPOSAL",
                "canonical": None,
                "method": "not-evaluated",
                "evidence": {
                    "reason": (
                        f"occurrences < {PROPOSAL_MIN_OCCURRENCES} or "
                        f"corpus_df_active > {PROPOSAL_MAX_DF}"
                    )
                },
                "alternatives": [],
            }
        proposal["confidence_tier"] = PROPOSAL_TIERS.get(proposal["method"], "none")
        target = proposal.get("canonical")
        if target:
            target_df = df_active.get(target, 0)
            proposal["bridge_gap_ratio"] = round(target_df / (active_df + 1), 6)
        record["canonicalization_proposal"] = proposal
        tokens.append(record)

    tokens.sort(key=lambda entry: (-entry["rank_score"], -entry["occurrences"], entry["token"]))

    oov_tokens = [t for t in tokens if t["is_oov"]]
    by_method = Counter(t["canonicalization_proposal"]["method"] for t in tokens)
    by_tier = Counter(t["canonicalization_proposal"]["confidence_tier"] for t in tokens)
    bridges = [
        t
        for t in tokens
        if t["canonicalization_proposal"]["confidence_tier"] == "strong"
        and t["canonicalization_proposal"].get("bridge_gap_ratio", 0) >= 2.0
    ]
    df_disagreements = [
        t for t in tokens if t["fts_df_active"] >= 0 and t["fts_df_active"] != t["corpus_df_active"]
    ]
    summary = {
        "distinct_tokens": len(tokens),
        "total_occurrences": sum(t["occurrences"] for t in tokens),
        "distinct_tokens_before_stoplist": len(tokens) + len(scan.stoplisted_tokens),
        "stoplisted_distinct_tokens": len(scan.stoplisted_tokens),
        "stoplisted_occurrences": sum(scan.stoplisted_tokens.values()),
        "oov_token_count": len(oov_tokens),
        "oov_occurrences": sum(t["occurrences"] for t in oov_tokens),
        "df_band_histogram": dict(sorted(Counter(t["df_band"] for t in tokens).items())),
        "proposal_method_histogram": dict(sorted(by_method.items())),
        "proposal_tier_histogram": dict(sorted(by_tier.items())),
        "session_attribution": {
            "events_scanned": scan.event_count,
            "events_without_session_id": scan.events_without_session,
            "note": (
                "distinct_sessions undercounts: most recall_events carry neither "
                "session_id nor transport_session_id"
            ),
        },
        "content_vs_fts_df": {
            "tokens_compared": len(tokens),
            "tokens_disagreeing": len(df_disagreements),
            "note": (
                "content scan counts active nodes whose text contains the token under "
                "this script's normalization; fts_df_active is the real bm25 path "
                "(fts5 unicode61). Both are measured over the SAME as-of-sliced active "
                "node set, so the remaining disagreements are tokenizer differences "
                "only -- not a node-set difference."
            ),
        },
        "cross_language_bridge_count": len(bridges),
        "cross_language_bridge_occurrences": sum(t["occurrences"] for t in bridges),
        "top_cross_language_bridges": [
            {
                "token": t["token"],
                "occurrences": t["occurrences"],
                "corpus_df_active": t["corpus_df_active"],
                "proposed": t["canonicalization_proposal"]["canonical"],
                "method": t["canonicalization_proposal"]["method"],
                "bridge_gap_ratio": t["canonicalization_proposal"]["bridge_gap_ratio"],
            }
            for t in sorted(bridges, key=lambda e: (-e["occurrences"], e["token"]))[:60]
        ],
    }
    return {
        "report": "oov-jargon",
        "provenance": provenance,
        "method": {
            "extraction": (
                "surface tokens of recall_events.query under the shipped "
                "normalization (embeddings.tokenize minus stemming/synonyms, which "
                "are latin-only and therefore identity on Cyrillic)"
            ),
            "token_filter": f"^[а-я]+$ after yo-folding, length >= {MIN_TOKEN_LENGTH}",
            "stoplist_module": module_stopwords,
            "stoplist_extra": extra_stopwords,
            "oov_definition": (
                "corpus_df_active == 0 AND fts_df_active == 0 -- absent from the "
                "content of every active node and unreachable through the bm25 "
                "FTS path (storage.py:1208-1227 joins nodes_fts with n.decayed = 0)"
            ),
            "ranking_formula": "rank_score = occurrences * 1/(1 + corpus_df_active)",
            "proposal_gate": (
                f"occurrences >= {PROPOSAL_MIN_OCCURRENCES} and "
                f"corpus_df_active <= {PROPOSAL_MAX_DF}"
            ),
            "proposal_disclaimer": (
                "EVERY canonicalization entry is a PROPOSAL. This script does not "
                "edit tokenize()/_SYNONYMS, the bm25/FTS path or the trigger channel."
            ),
        },
        "summary": summary,
        "tokens": tokens,
    }


# --------------------------------------------------------------------------
# Provenance.
# --------------------------------------------------------------------------
def build_provenance(
    conn: sqlite3.Connection, db_path: Path, as_of: str, *, as_of_in_future: bool
) -> dict[str, Any]:
    stat = db_path.stat()
    counts = {
        "nodes_total": conn.execute("SELECT count(*) FROM nodes").fetchone()[0],
        "nodes_active": conn.execute("SELECT count(*) FROM nodes WHERE decayed = 0").fetchone()[0],
        "nodes_in_slice": conn.execute(
            "SELECT count(*) FROM nodes WHERE created_at <= :as_of", {"as_of": as_of}
        ).fetchone()[0],
        "schema_nodes_active_in_slice": conn.execute(
            "SELECT count(*) FROM nodes WHERE decayed = 0 AND level = 'schema' "
            "AND created_at <= :as_of",
            {"as_of": as_of},
        ).fetchone()[0],
        "recall_events_total": conn.execute("SELECT count(*) FROM recall_events").fetchone()[0],
        "recall_events_in_slice": conn.execute(
            "SELECT count(*) FROM recall_events WHERE created_at <= :as_of", {"as_of": as_of}
        ).fetchone()[0],
    }
    return {
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "generated_at": as_of,
        "as_of": as_of,
        "as_of_meaning": (
            "single knob: generation timestamp AND slice cutoff "
            "(created_at <= as_of on both nodes and recall_events). Re-run with "
            "--as-of <this value> to reproduce this slice."
        ),
        "as_of_in_future": as_of_in_future,
        "slice_frozen": not as_of_in_future,
        "database": {
            "path": str(db_path),
            "bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        },
        "access": (
            "sqlite3.connect(file:<path>?mode=ro, uri=True) + PRAGMA query_only=1 + one "
            "deferred read transaction pinning a single WAL snapshot; MemoryStore is "
            "never imported (its constructor migrates and writes)"
        ),
        "row_counts": counts,
        "query_slices": {
            "schemas": SQL_SCHEMAS,
            "recall_events": SQL_EVENTS,
            "node_content_for_corpus_df": SQL_NODE_CONTENT,
            "fts_document_frequency": SQL_FTS_DF_ACTIVE,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "sqlite": sqlite3.sqlite_version,
        },
        "git": git_provenance(),
        "determinism": DETERMINISM_CONTRACT,
        "caveats": [
            "Determinism has THREE layers, not two. (1) The derived analysis -- token "
            "set, occurrence counts, event-slice statistics, trigger firing, "
            "collisions, canonicalization proposals -- is frozen by --as-of and is "
            "byte-identical across re-runs. (2) A handful of per-node columns are "
            "CURRENT STATE AT READ, not as-of state, because SQLite stores only their "
            "latest value and no history: they move whenever the live server serves a "
            "recall. They are enumerated field-by-field in "
            "`provenance.determinism.current_state_at_read`. (3) The PROVENANCE block "
            "deliberately describes the live file at read time, so database "
            "bytes/mtime, the unsliced *_total counts and the git dirty flag move too. "
            "Run `python3 scripts/handoff_inventory.py --verify` to check this "
            "contract mechanically: it fails if any layer-1 value drifted.",
            "decayed = 0 is CURRENT state, not as-of state: the node set can shift "
            "if a schema decays after this run, while the event slice cannot.",
            "recall_events.results records only DELIVERED results (max_results cap), "
            "so a trigger that matched but lost its slot leaves no trace there -- "
            "that is what the simulated_trigger_matches column measures.",
            "The live database is written concurrently by the MCP server; the read "
            "transaction isolates one snapshot, and --as-of freezes the slice.",
        ],
    }


# --------------------------------------------------------------------------
# Markdown rendering.
# --------------------------------------------------------------------------
def md_escape(text: Any) -> str:
    value = "" if text is None else str(text)
    return value.replace("|", "\\|").replace("\n", " ").strip()


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(md_escape(cell) for cell in row) + " |")
    if not rows:
        lines.append("| " + " | ".join("--" for _ in headers) + " |")
    return lines


def md_provenance(provenance: dict[str, Any]) -> list[str]:
    git = provenance["git"]
    counts = provenance["row_counts"]
    lines = [
        "## Provenance",
        "",
        f"- generator: `{provenance['generator']}` v{provenance['generator_version']}",
        f"- generated_at / slice cutoff (`--as-of`): `{provenance['as_of']}` "
        f"(slice frozen: {provenance['slice_frozen']})",
        f"- database: `{provenance['database']['path']}` "
        f"({provenance['database']['bytes']} bytes, mtime {provenance['database']['mtime_utc']})",
        f"- access: {provenance['access']}",
        f"- git commit: `{git['commit']}` (branch `{git['branch']}`, "
        f"worktree dirty: {git['worktree_dirty']}) -- rebase-fragile, see below",
        f"- generator sha256: `{git['generator_sha256']}` "
        "(rebase-invariant anchor; `sha256sum scripts/handoff_inventory.py`)",
        f"- runtime: python {provenance['runtime']['python']}, "
        f"sqlite {provenance['runtime']['sqlite']}",
        "- row counts: "
        + ", ".join(f"{key} = {value}" for key, value in sorted(counts.items())),
        "",
        "### Exact query slices",
        "",
        "```sql",
    ]
    for name, sql in sorted(provenance["query_slices"].items()):
        lines.append(f"-- {name}")
        lines.append(sql + ";")
    lines.append("```")
    determinism = provenance["determinism"]
    lines.append("")
    lines.append("### Determinism contract")
    lines.append("")
    lines.append(f"- frozen by `--as-of`: {determinism['frozen_by_as_of']}")
    lines.append(
        "- current state at read (NOT reproducible, tolerated by `--verify`): "
        + ", ".join(f"`{field}`" for field in determinism["current_state_at_read"])
    )
    lines.append(f"- why: {determinism['why_not_frozen']}")
    lines.append(f"- decay: {determinism['why_decay_is_NOT_tolerated']}")
    lines.append(f"- generator binding: {determinism['generator_binding']}")
    lines.append("")
    lines.append("### Caveats")
    lines.append("")
    lines.extend(f"- {caveat}" for caveat in provenance["caveats"])
    lines.append("")
    lines.append(
        f"Reproduce: `python3 {GENERATOR} --db {provenance['database']['path']} "
        f"--as-of {provenance['as_of']}`"
    )
    lines.append(f"Verify instead of rewrite: `{determinism['verify_command']}`")
    lines.append("")
    return lines


def render_schema_markdown(report: dict[str, Any], md_top: int) -> str:
    summary = report["summary"]
    delivery = summary["delivery"]
    simulation = summary["simulation"]
    schemas = report["schemas"]
    by_id = {schema["id"]: schema for schema in schemas}
    rule = report["trigger_rule"]

    lines = [
        "# Handoff inventory: active `level:schema` nodes and their triggers",
        "",
        "Read-only evidence for the *when_to_use triggers* goal. Nothing here was "
        "written back to the database and no source module was modified.",
        "",
    ]
    lines += md_provenance(report["provenance"])
    lines += [
        "## The rule being measured",
        "",
        f"- source: `{rule['source']}`",
        f"- `{rule['formula']}`",
        f"- fires when `overlap >= {rule['overlap_threshold']}`, "
        f"scoring `{rule['base_score']} + 0.05 * overlap`",
        "- the channel only *adds a candidate*; whether that candidate is delivered "
        "depends on the final ranking and on `max_results`.",
        "",
        "## Population",
        "",
        f"- active schema nodes in slice: **{summary['active_schema_count']}** "
        f"(all {summary['with_trigger']} carry `context.trigger`, "
        f"{summary['with_trigger_tokens']} tokenize to a non-empty trigger set)",
        f"- with `task_pattern`: {summary['with_task_pattern']}; "
        f"with `procedure_id`: {summary['with_procedure_id']}",
        f"- distinct trigger texts: **{summary['distinct_trigger_texts']}** "
        f"(distinct token sets: {summary['distinct_trigger_token_sets']}) -- "
        f"{summary['active_schema_count'] - summary['distinct_trigger_texts']} schemas "
        "repeat a trigger that already exists",
        f"- content size: min {summary['content_chars']['min']}, "
        f"median {summary['content_chars']['median']}, max {summary['content_chars']['max']} chars",
        "",
        "### Scope distribution",
        "",
    ]
    lines += md_table(
        ["scope", "schemas"],
        [[scope, count] for scope, count in sorted(summary["scopes"].items(), key=lambda kv: (-kv[1], kv[0]))],
    )
    lines += [
        "",
        "### Trigger token-count histogram",
        "",
    ]
    lines += md_table(
        ["trigger tokens", "schemas"],
        [[count, n] for count, n in sorted(summary["trigger_token_count_histogram"].items())],
    )
    lines += [
        "",
        "## Firing: what actually surfaced",
        "",
        f"- schema deliveries recorded in `recall_events.results`: "
        f"**{delivery['total_schema_deliveries']}**",
        f"- of those, trigger-channel hits (`trigger_score > 0`): "
        f"**{delivery['total_trigger_channel_hits']}**; delivered by other channels only: "
        f"**{delivery['total_other_channel_hits']}**",
        f"- schemas ever delivered: **{delivery['delivered_schema_count']}** / "
        f"{summary['active_schema_count']}; **never delivered: "
        f"{delivery['never_delivered_count']}**",
        f"- never trigger-fired: {delivery['never_trigger_fired_count']}; "
        f"delivered only via bm25/vector/graph: "
        f"{delivery['delivered_only_via_other_channels_count']}",
        "",
        "Channel mix over delivered schema results (a result can carry several channels):",
        "",
    ]
    lines += md_table(
        ["channel", "delivered schema results"],
        [
            [channel, count]
            for channel, count in sorted(
                delivery["channel_counts"].items(), key=lambda kv: (-kv[1], kv[0])
            )
        ],
    )
    lines += [
        "",
        "## Firing: what the trigger rule *would* have matched",
        "",
        f"{simulation['what_it_is']}.",
        "",
        f"- simulated trigger matches: **{simulation['total_simulated_matches']}** vs "
        f"**{delivery['total_trigger_channel_hits']}** actually delivered "
        f"({_percent(delivery['total_trigger_channel_hits'], simulation['total_simulated_matches'])} "
        "of matches reached the caller)",
        f"- schemas that would match at least once: {simulation['matched_schema_count']}; "
        f"never matched: **{simulation['never_matched_count']}**",
        f"- schemas that matched but were never delivered: "
        f"**{simulation['matched_but_never_delivered_count']}**",
        f"- matches in events where more schemas fired than `max_results` could return: "
        f"**{simulation['total_crowded_matches']}**",
        "",
        "Simultaneous trigger matches per event (how crowded the channel gets):",
        "",
    ]
    lines += md_table(
        ["schemas matching one query", "events"],
        [[count, n] for count, n in sorted(simulation["matches_per_event_histogram"].items())],
    )
    lines += [
        "",
        f"## Top {min(md_top, len(summary['top_firers']))} firers",
        "",
    ]
    lines += md_table(
        [
            "id",
            "scope",
            "trigger",
            "delivered",
            "trigger hits",
            "simulated matches",
            "delivered share",
        ],
        [
            [
                entry["id"],
                entry["scope"],
                entry["trigger"],
                entry["delivered_events"],
                entry["trigger_channel_hits"],
                entry["simulated_trigger_matches"],
                _percent(entry["trigger_channel_hits"], entry["simulated_trigger_matches"]),
            ]
            for entry in summary["top_firers"][:md_top]
        ],
    )

    never = [by_id[node_id] for node_id in summary["never_delivered_ids"]]
    never.sort(key=lambda n: (-n["firing"]["simulated_trigger_matches"], n["id"]))
    lines += [
        "",
        f"## Never delivered ({len(never)} schemas)",
        "",
        f"Split by cause -- the actionable distinction for the trigger goal: "
        f"**{simulation['matched_but_never_delivered_count']}** of these DID match the "
        f"trigger rule and still never reached a caller (they need less competition), "
        f"while **{len(never) - simulation['matched_but_never_delivered_count']}** never "
        "matched any recorded query at all (they need a better trigger).",
        "",
    ]
    lines += md_table(
        ["id", "scope", "trigger", "trigger tokens", "simulated matches", "eligible events"],
        [
            [
                n["id"],
                n["scope"],
                n["trigger"],
                n["trigger_token_count"],
                n["firing"]["simulated_trigger_matches"],
                n["firing"]["eligible_events"],
            ]
            for n in never[:md_top]
        ],
    )
    if len(never) > md_top:
        lines.append("")
        lines.append(
            f"_{len(never) - md_top} further never-delivered schemas are in the JSON "
            "under `summary.never_delivered_ids`._"
        )

    duplicates = summary["duplicate_trigger_texts"]
    lines += [
        "",
        "## Trigger duplication",
        "",
        f"{len(duplicates)} trigger texts are shared by more than one schema; they "
        f"account for {sum(entry['count'] for entry in duplicates)} of "
        f"{summary['active_schema_count']} schemas. Duplicates fire together by "
        "construction and then compete for the same result slots.",
        "",
    ]
    lines += md_table(
        ["trigger", "schemas", "scopes"],
        [
            [
                entry["trigger"],
                entry["count"],
                ", ".join(f"{scope}:{count}" for scope, count in entry["scopes"].items()),
            ]
            for entry in duplicates[:md_top]
        ],
    )

    collisions = summary["collisions"]
    lines += [
        "",
        "## Trigger-token collisions",
        "",
        f"Collision rule: `{collisions['rule']}`.",
        "",
        f"- colliding pairs: **{collisions['pair_count']}**",
        f"- schemas involved in at least one collision: "
        f"**{collisions['colliding_schema_count']}** / {summary['active_schema_count']}",
        f"- collision clusters (connected components): {collisions['cluster_count']}",
        f"- groups with byte-identical trigger token sets: "
        f"{len(collisions['identical_trigger_token_sets'])}",
        "",
        "Largest clusters:",
        "",
    ]
    lines += md_table(
        ["size", "scopes", "example triggers"],
        [
            [
                cluster["size"],
                ", ".join(
                    sorted({by_id[member]["scope"] for member in cluster["members"]})
                ),
                " / ".join(
                    sorted({by_id[member]["trigger"] for member in cluster["members"]})[:4]
                ),
            ]
            for cluster in collisions["clusters"][:md_top]
        ],
    )

    hair = summary["hair_triggers"]
    lines += [
        "",
        f"## Hair triggers ({len(hair)} schemas with 1-2 trigger tokens)",
        "",
        "With `overlap >= 0.5`, a two-token trigger fires on ONE shared token and a "
        "one-token trigger fires on any query containing that token.",
        "",
    ]
    lines += md_table(
        ["id", "scope", "trigger", "tokens", "simulated matches", "trigger hits"],
        [
            [
                entry["id"],
                entry["scope"],
                entry["trigger"],
                ", ".join(entry["trigger_tokens"]),
                entry["simulated_trigger_matches"],
                entry["trigger_channel_hits"],
            ]
            for entry in hair[:md_top]
        ],
    )
    lines += [
        "",
        "## Full inventory",
        "",
        "`artifacts/handoff/schema-inventory.json` carries every active schema with "
        "`id, scope, task_pattern, procedure_id, trigger, trigger_tokens, "
        "content_chars, created_at, access_count, usefulness_score` and the full "
        "`firing` block (delivered/trigger/other-channel counts, channel mix, best "
        "rank, distinct sessions/scopes/queries, eligible events, simulated matches).",
        "",
    ]
    return "\n".join(lines) + "\n"


def _percent(part: int, whole: int) -> str:
    if not whole:
        return "n/a"
    return f"{100.0 * part / whole:.1f}%"


def render_jargon_markdown(report: dict[str, Any], md_top: int) -> str:
    summary = report["summary"]
    method = report["method"]
    tokens = report["tokens"]

    lines = [
        "# Handoff inventory: Cyrillic query jargon and its corpus coverage",
        "",
        "Read-only evidence for the *jargon dictionary* goal. Every canonicalization "
        "below is a **PROPOSAL**: this run did not touch `tokenize()`, `_SYNONYMS`, "
        "the bm25/FTS path or any source module.",
        "",
    ]
    lines += md_provenance(report["provenance"])
    lines += [
        "## Method",
        "",
        f"- extraction: {method['extraction']}",
        f"- token filter: `{method['token_filter']}`",
        f"- OOV definition: {method['oov_definition']}",
        f"- ranking: `{method['ranking_formula']}` -- a token nobody stored keeps its "
        "full query frequency, a token stored in N active nodes keeps `1/(1+N)` of it",
        f"- proposals computed when: `{method['proposal_gate']}`",
        "",
        "### Stoplist (explicit)",
        "",
        f"Shipped Russian stop words from `living_memory.embeddings._RUSSIAN_STOP_WORDS` "
        f"with length >= {MIN_TOKEN_LENGTH} ({len(method['stoplist_module'])} entries):",
        "",
        "> " + ", ".join(method["stoplist_module"]),
        "",
        f"Plus this script's explicit `EXTRA_CYRILLIC_STOPWORDS` "
        f"({len(method['stoplist_extra'])} entries), function words that survive the "
        "shipped list but carry no retrieval signal:",
        "",
        "> " + ", ".join(method["stoplist_extra"]),
        "",
        "## Population",
        "",
        f"- distinct Cyrillic query tokens (length >= {MIN_TOKEN_LENGTH}) before the "
        f"stoplist: {summary['distinct_tokens_before_stoplist']}; the stoplist removes "
        f"{summary['stoplisted_distinct_tokens']} of them "
        f"({summary['stoplisted_occurrences']} occurrences)",
        f"- distinct Cyrillic query tokens after the stoplist: "
        f"**{summary['distinct_tokens']}** over {summary['total_occurrences']} occurrences",
        f"- strictly out-of-vocabulary (absent from every active node AND from the "
        f"active-node FTS index): **{summary['oov_token_count']}** tokens, "
        f"{summary['oov_occurrences']} occurrences",
        f"- cross-language bridge candidates (a latin form exists in the corpus with "
        f"at least 2x the Cyrillic token's coverage): "
        f"**{summary['cross_language_bridge_count']}** tokens, "
        f"{summary['cross_language_bridge_occurrences']} occurrences",
        "",
        "Corpus coverage bands (`corpus_df_active` = active nodes whose content "
        "contains the token):",
        "",
    ]
    band_order = ["0", "1", "2-5", "6-20", "21-100", ">100"]
    lines += md_table(
        ["corpus_df_active", "tokens"],
        [
            [band, summary["df_band_histogram"][band]]
            for band in band_order
            if band in summary["df_band_histogram"]
        ],
    )
    lines += [
        "",
        "Proposal methods and their evidence tier (`strong` = curated or an exact "
        "transliteration hit in the corpus; `medium` = identical consonant skeleton; "
        "`weak` = bounded edit distance or a prefix completion -- JSON-only material, "
        "not quoted as a bridge below):",
        "",
    ]
    lines += md_table(
        ["method", "tier", "tokens"],
        [
            [name, PROPOSAL_TIERS.get(name, "none"), count]
            for name, count in sorted(
                summary["proposal_method_histogram"].items(), key=lambda kv: (-kv[1], kv[0])
            )
        ],
    )
    session = summary["session_attribution"]
    fts = summary["content_vs_fts_df"]
    lines += [
        "",
        f"- session attribution: {session['events_without_session_id']} of "
        f"{session['events_scanned']} events carry neither `session_id` nor "
        f"`transport_session_id`, so `distinct_sessions` is a lower bound",
        f"- content scan vs FTS index: {fts['tokens_disagreeing']} of "
        f"{fts['tokens_compared']} tokens disagree between `corpus_df_active` and "
        f"`fts_df_active` ({fts['note']})",
    ]
    lines += [
        "",
        f"## Top {md_top} by `rank_score` (frequency x OOV-ness)",
        "",
        "This is the list to cut a threshold on.",
        "",
    ]
    lines += md_table(
        [
            "token",
            "occ",
            "queries",
            "sessions",
            "scopes",
            "df_active",
            "fts_df",
            "OOV",
            "rank_score",
            "PROPOSAL",
            "method",
            "tier",
        ],
        [
            [
                entry["token"],
                entry["occurrences"],
                entry["distinct_queries"],
                entry["distinct_sessions"],
                entry["distinct_scopes"],
                entry["corpus_df_active"],
                entry["fts_df_active"],
                "yes" if entry["is_oov"] else "no",
                f"{entry['rank_score']:.2f}",
                entry["canonicalization_proposal"]["canonical"] or "--",
                entry["canonicalization_proposal"]["method"],
                entry["canonicalization_proposal"]["confidence_tier"],
            ]
            for entry in tokens[:md_top]
        ],
    )
    actionable = [
        entry
        for entry in tokens
        if entry["canonicalization_proposal"]["canonical"]
        and entry["canonicalization_proposal"]["confidence_tier"] in {"strong", "medium"}
    ]
    lines += [
        "",
        f"## Top {min(md_top, len(actionable))} actionable candidates "
        f"(strong/medium evidence, ranked the same way)",
        "",
        "The same ranking restricted to tokens that actually have a defensible "
        "canonical form. `in-corpus` tokens are dropped here: bm25 already reaches "
        "them, so they need no synonym.",
        "",
    ]
    lines += md_table(
        ["token", "occ", "df_active", "OOV", "rank_score", "PROPOSAL", "method", "gap"],
        [
            [
                entry["token"],
                entry["occurrences"],
                entry["corpus_df_active"],
                "yes" if entry["is_oov"] else "no",
                f"{entry['rank_score']:.2f}",
                entry["canonicalization_proposal"]["canonical"],
                entry["canonicalization_proposal"]["method"],
                f"{entry['canonicalization_proposal'].get('bridge_gap_ratio', 0):.1f}x",
            ]
            for entry in actionable[:md_top]
        ],
    )
    lines += [
        "",
        "## Cross-language bridge candidates (strong tier only)",
        "",
        "The class the root goal named: a Russian-spelled borrowing whose English "
        "form is what the corpus actually stores. `gap` = "
        "`df(proposed latin form) / (1 + df(cyrillic token))`. Restricted to curated "
        "and exact-transliteration evidence -- weaker matches live in the JSON.",
        "",
    ]
    lines += md_table(
        ["token", "occ", "df_active", "PROPOSAL", "method", "gap"],
        [
            [
                entry["token"],
                entry["occurrences"],
                entry["corpus_df_active"],
                entry["proposed"],
                entry["method"],
                f"{entry['bridge_gap_ratio']:.1f}x",
            ]
            for entry in summary["top_cross_language_bridges"][:md_top]
        ],
    )
    truncated = [
        entry
        for entry in tokens
        if entry["canonicalization_proposal"]["method"] == "cyrillic-prefix"
    ][:md_top]
    lines += [
        "",
        "## Truncated / misspelled query tokens",
        "",
        "Tokens whose only in-corpus match is a word they are a strict prefix of "
        "(queries cut mid-word) or a near-identical spelling. These are *not* jargon "
        "and should probably be excluded from a synonym table rather than mapped.",
        "",
    ]
    lines += md_table(
        ["token", "occ", "df_active", "nearest in-corpus word", "method"],
        [
            [
                entry["token"],
                entry["occurrences"],
                entry["corpus_df_active"],
                entry["canonicalization_proposal"]["canonical"],
                entry["canonicalization_proposal"]["method"],
            ]
            for entry in truncated
        ],
    )
    lines += [
        "",
        "## Full dictionary",
        "",
        f"`artifacts/handoff/oov-jargon.json` carries all {summary['distinct_tokens']} "
        "tokens sorted by `rank_score`, each with occurrences, distinct "
        "queries/events/sessions/scopes, top scopes, `corpus_df_active`, "
        "`corpus_df_all_nodes`, `fts_df_active`, the OOV flag and the "
        "`canonicalization_proposal` block (canonical, method, evidence, "
        "alternatives, bridge_gap_ratio).",
        "",
        f"**{method['proposal_disclaimer']}**",
        "",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------
def serialize_payload(payload: dict[str, Any]) -> str:
    """The one JSON encoding used for both writing and verifying."""

    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_outputs(out_dir: Path, stem: str, payload: dict[str, Any], markdown: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(serialize_payload(payload), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    return [json_path, md_path]


def build_reports(
    conn: sqlite3.Connection,
    db_path: Path,
    as_of: str,
    *,
    as_of_in_future: bool,
    which: str,
    md_top: int,
) -> list[tuple[str, dict[str, Any], str]]:
    """Build the requested reports.

    Single source of truth for both writing and ``--verify``: the verifier must
    exercise the same code path it is checking, or it proves nothing.
    """

    provenance = build_provenance(conn, db_path, as_of, as_of_in_future=as_of_in_future)
    stoplist, module_stopwords, extra_stopwords = build_stoplist()
    schemas = load_schemas(conn, as_of)
    scan = EventScan(schemas, stoplist)
    scan.run(conn, as_of)
    provenance["row_counts"]["recall_events_scanned"] = scan.event_count
    provenance["row_counts"]["result_rows_scanned"] = scan.result_rows
    provenance["row_counts"]["first_event_at"] = scan.first_event_at
    provenance["row_counts"]["last_event_at"] = scan.last_event_at

    reports: list[tuple[str, dict[str, Any], str]] = []
    if which in {"all", "schema"}:
        # deepcopy so each report embeds the provenance as it stood for THAT
        # report -- otherwise --report all and --report schema would disagree.
        report = build_schema_report(
            schemas, scan, find_collisions(schemas), copy.deepcopy(provenance)
        )
        reports.append(("schema-inventory", report, render_schema_markdown(report, md_top)))
    if which in {"all", "jargon"}:
        df_active, df_all, active_nodes, all_nodes = build_corpus_df(conn, as_of)
        provenance["row_counts"]["corpus_nodes_active"] = active_nodes
        provenance["row_counts"]["corpus_nodes_all"] = all_nodes
        provenance["row_counts"]["corpus_distinct_surface_tokens"] = len(df_all)
        report = build_jargon_report(
            conn,
            scan,
            df_active,
            df_all,
            module_stopwords,
            extra_stopwords,
            copy.deepcopy(provenance),
        )
        reports.append(("oov-jargon", report, render_jargon_markdown(report, md_top)))
    return reports


# --------------------------------------------------------------------------
# --verify: make the determinism claim executable instead of asserted.
# --------------------------------------------------------------------------
_VOLATILE_SCHEMA_RE = re.compile(
    r"^schemas\[\d+\]\.(?:" + "|".join(VOLATILE_SCHEMA_FIELDS) + r")$"
)


def diff_leaves(
    committed: Any, regenerated: Any, path: str = ""
) -> Iterable[tuple[str, Any, Any]]:
    """Yield ``(path, committed, regenerated)`` for every differing leaf."""

    if isinstance(committed, dict) and isinstance(regenerated, dict):
        for key in sorted(set(committed) | set(regenerated)):
            sub = f"{path}.{key}" if path else key
            if key not in committed:
                yield sub, "<absent>", regenerated[key]
            elif key not in regenerated:
                yield sub, committed[key], "<absent>"
            else:
                yield from diff_leaves(committed[key], regenerated[key], sub)
    elif isinstance(committed, list) and isinstance(regenerated, list):
        if len(committed) != len(regenerated):
            yield f"{path}[]", f"<{len(committed)} items>", f"<{len(regenerated)} items>"
        else:
            for index, (left, right) in enumerate(zip(committed, regenerated)):
                yield from diff_leaves(left, right, f"{path}[{index}]")
    elif committed != regenerated:
        yield path or "<root>", committed, regenerated


def is_tolerated_drift(path: str) -> bool:
    """True for values the determinism contract declares current-state-at-read."""

    if _VOLATILE_SCHEMA_RE.match(path):
        return True
    if path.startswith("provenance."):
        return path[len("provenance.") :] in VOLATILE_PROVENANCE_PATHS
    return False


def recorded_as_of(out_dir: Path) -> str | None:
    for stem in ("schema-inventory", "oov-jargon"):
        path = out_dir / f"{stem}.json"
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))["provenance"]["as_of"]
        except (ValueError, KeyError):  # pragma: no cover - corrupt artifact
            continue
    return None


def verify_reports(out_dir: Path, reports: list[tuple[str, dict[str, Any], str]]) -> int:
    """Compare freshly built reports against the committed ones. 0 = contract holds."""

    failures = 0
    for stem, payload, _markdown in reports:
        path = out_dir / f"{stem}.json"
        if not path.exists():
            print(f"VERIFY {stem}: FAIL -- committed report missing at {path}", file=sys.stderr)
            failures += 1
            continue
        on_disk = path.read_text(encoding="utf-8")
        rendered = serialize_payload(payload)
        if on_disk == rendered:
            print(f"VERIFY {stem}: OK -- byte-identical to the committed report")
            continue
        committed = json.loads(on_disk)
        # Round-trip the fresh payload through the same encoder so the comparison
        # sees what WOULD be written (JSON coerces int dict keys to strings).
        regenerated = json.loads(rendered)
        frozen: list[tuple[str, Any, Any]] = []
        tolerated: list[tuple[str, Any, Any]] = []
        for item in diff_leaves(committed, regenerated):
            (tolerated if is_tolerated_drift(item[0]) else frozen).append(item)
        status = "OK" if not frozen else "FAIL"
        print(
            f"VERIFY {stem}: {status} -- {len(frozen)} frozen drift, "
            f"{len(tolerated)} tolerated current-state drift"
        )
        for diff_path, old, new in frozen[:20]:
            print(f"    FROZEN DRIFT {diff_path}: committed={old!r} regenerated={new!r}")
        if len(frozen) > 20:
            print(f"    ... and {len(frozen) - 20} more")
        if frozen:
            failures += 1
    return 1 if failures else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine read-only handoff inventories (level:schema triggers + Cyrillic "
            "query jargon) from a Living Memory database."
        )
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "rebuild in memory and compare against the committed reports instead of "
            "rewriting them; exits 1 if any value the determinism contract calls "
            "frozen has drifted. Defaults --as-of to the recorded one."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"database to read, opened mode=ro (default: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"directory for the reports (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help=(
            "ISO-8601 Z timestamp used as BOTH the generation timestamp and the "
            "slice cutoff (created_at <= as-of). Default: now (UTC). Pass the value "
            "recorded in a report's provenance to reproduce it."
        ),
    )
    parser.add_argument(
        "--report",
        choices=["all", "schema", "jargon"],
        default="all",
        help="which report(s) to regenerate (default: all)",
    )
    parser.add_argument(
        "--md-top",
        type=int,
        default=40,
        help="rows per Markdown table; the JSON always holds the full data (default: 40)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    as_of = args.as_of
    if as_of is None and args.verify:
        # Verifying means "reproduce what is committed", so the recorded cutoff is
        # the only meaningful one; falling back to now() would compare two slices.
        as_of = recorded_as_of(args.out_dir)
        if as_of is None:
            raise SystemExit(
                f"--verify found no committed report with a recorded as_of in {args.out_dir}; "
                "pass --as-of explicitly"
            )
        print(f"--verify: reusing recorded as_of {as_of}", file=sys.stderr)
    as_of = as_of or datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", as_of):
        raise SystemExit(f"--as-of must look like 2026-08-17T12:00:00Z, got: {as_of}")

    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    as_of_in_future = as_of > now
    if as_of_in_future:
        # A cutoff ahead of wall clock freezes nothing: rows written between now
        # and the cutoff still enter the slice on the next run.
        print(
            f"WARNING: --as-of {as_of} is in the future (now {now}); the slice is "
            "NOT frozen and a later run will pick up rows written in between.",
            file=sys.stderr,
        )

    db_path = args.db.expanduser().resolve()
    conn = open_readonly(db_path)
    try:
        reports = build_reports(
            conn,
            db_path,
            as_of,
            as_of_in_future=as_of_in_future,
            which=args.report,
            md_top=args.md_top,
        )
    finally:
        conn.rollback()
        conn.close()

    if args.verify:
        return verify_reports(args.out_dir, reports)

    written: list[Path] = []
    for stem, payload, markdown in reports:
        written += write_outputs(args.out_dir, stem, payload, markdown)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
