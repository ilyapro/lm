"""De-identification engine for the animal-planet replay corpus.

Every private text field (node content, context string values, query text,
task labels, agent names, session labels, correction old/new texts) is
replaced by a surrogate that preserves exactly three invariants:

(a) exact character length (``len`` in Python characters);
(b) whitespace positions — characters in ``KEEP_WS`` (space, tab, newline,
    carriage return) are kept verbatim at their positions; every other
    character (including exotic whitespace such as U+00A0/U+2028 and control
    characters, so that the purity contract ``^[a-z \\t\\n\\r]*$`` holds)
    becomes a pseudo-random lowercase a-z letter;
(c) content-equality classes — identical originals map to identical
    surrogates, distinct originals to distinct surrogates, globally across
    the whole build (``living_memory.delivery.shape_recall_results`` twin
    dedup compares content strings byte-for-byte; snippeting and previews
    cut by character length and newline positions).

Surrogate letters derive from HMAC-SHA256 over the original string keyed by
an ephemeral salt generated at build time and never recorded anywhere, so
the mapping is non-invertible once the build process exits.  Collisions
between distinct originals (possible for very short strings: a 1-letter
original has only 26 surrogates) are resolved by re-deriving with an attempt
counter until the surrogate is unique; the resolution is deterministic
within one build (fixed processing order, fixed salt).

Kept real (never surrogated): ULIDs and other opaque identifiers, scope
names, ISO timestamps, scores, levels, flags, per-field character counts of
the originals, context key names, and recognized automatic-recall template
fingerprints (class + template_id per the staging METADATA predicates).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from typing import Any

# Whitespace kept verbatim; everything else becomes a lowercase letter.
KEEP_WS = frozenset(" \t\n\r")
PURITY_RE = re.compile(r"^[a-z \t\n\r]*$")

# Kept-real value classes for generic (context/ambient_context) string leaves.
ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")
HEX32_RE = re.compile(r"^[0-9a-f]{32}$")
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
# Hex ids (git SHAs etc.): require >=1 digit so all-letter English words
# spelled from a-f ("defaced") are not misclassified as identifiers.
HEXID_RE = re.compile(r"^(?=.*[0-9])[0-9a-f]{7,64}$")
ISO_TS_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$"
)
NUMERIC_RE = re.compile(r"^[0-9]+([.][0-9]+)?$")
FLAG_LITERALS = frozenset({"true", "false", "yes", "no"})
TEMPORAL_HINT_RE = re.compile(r"^[a-z]+(:[a-z0-9_-]+)?$")

# METADATA.predicates.deterministic_prerecalls: recognized automatic-recall
# templates, matched as substrings of the ORIGINAL query (fixed order).
TEMPLATE_IDS = ("reopen_lesson", "architectural_decision")


def classify_template(query: str) -> str | None:
    """Return the pinned template_id fingerprint for an original query."""

    for template_id in TEMPLATE_IDS:
        if template_id in query:
            return template_id
    return None


def classify_class(agent: str | None) -> str:
    """METADATA.predicates.organic_vs_automatic: automatic iff agent IS NULL."""

    return "automatic" if agent is None else "organic"


def keeps_real(value: str, known_scopes: frozenset[str] | set[str]) -> bool:
    """True when a generic string leaf belongs to a kept-real class."""

    return bool(
        value in known_scopes
        or ULID_RE.match(value)
        or HEX32_RE.match(value)
        or UUID_RE.match(value)
        or HEXID_RE.match(value)
        or ISO_TS_RE.match(value)
        or NUMERIC_RE.match(value)
        or value in FLAG_LITERALS
    )


def delivery_chars(value: Any) -> int:
    """Char count with living_memory.delivery semantics.

    Strings count as ``len``; lists/dicts (and any other JSON value) count as
    ``len(json.dumps(value, ensure_ascii=False, default=str))`` — mirroring
    ``delivery._json_chars`` / ``_summarize_context_value``.
    """

    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, ensure_ascii=False, default=str))


class SurrogatePool:
    """Global original->surrogate map for one build.

    Deterministic within one build: the caller feeds strings in a fixed
    order, the salt is fixed for the build, and collision resolution uses an
    attempt counter.  Nothing about the salt or the mapping is persisted by
    this class; byte-level corpus identity is guaranteed by the frozen
    tracked output files, not by re-running the transform.
    """

    def __init__(self, salt: bytes) -> None:
        self._salt = salt
        self._by_original: dict[str, str] = {}
        self._taken: dict[str, str] = {}  # surrogate -> original
        self.collision_retries = 0

    def _letters(self, original: str, attempt: int, count: int) -> list[str]:
        letters: list[str] = []
        seed = attempt.to_bytes(4, "big") + original.encode("utf-8")
        block_index = 0
        while len(letters) < count:
            digest = hmac.new(
                self._salt, seed + block_index.to_bytes(4, "big"), hashlib.sha256
            ).digest()
            letters.extend(chr(ord("a") + byte % 26) for byte in digest)
            block_index += 1
        return letters[:count]

    def _render(self, original: str, attempt: int) -> str:
        need = sum(1 for ch in original if ch not in KEEP_WS)
        letters = iter(self._letters(original, attempt, need))
        return "".join(ch if ch in KEEP_WS else next(letters) for ch in original)

    def surrogate(self, original: str) -> str:
        existing = self._by_original.get(original)
        if existing is not None:
            return existing
        attempt = 0
        candidate = self._render(original, attempt)
        while candidate in self._taken and self._taken[candidate] != original:
            attempt += 1
            self.collision_retries += 1
            candidate = self._render(original, attempt)
        self._by_original[original] = candidate
        self._taken[candidate] = original
        return candidate

    def mapping(self) -> dict[str, str]:
        """The full original->surrogate map (private; staging-only output)."""

        return dict(self._by_original)


def transform_value(
    value: Any, pool: SurrogatePool, known_scopes: frozenset[str] | set[str]
) -> Any:
    """Recursively de-identify a JSON value: keys kept, string leaves either
    kept real (per :func:`keeps_real`) or surrogated; non-strings kept."""

    if isinstance(value, dict):
        return {
            key: transform_value(item, pool, known_scopes)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [transform_value(item, pool, known_scopes) for item in value]
    if isinstance(value, str):
        if keeps_real(value, known_scopes):
            return value
        return pool.surrogate(value)
    return value


def context_chars_map(context: dict[str, Any]) -> dict[str, int]:
    """Per-top-level-key char counts of the ORIGINAL context values."""

    return {key: delivery_chars(value) for key, value in context.items()}


def provenance_shape_of(
    provenance: dict[str, Any], source_traces: list, corrections: list
) -> dict[str, Any]:
    """Shape/sizes of a node's provenance, never its text.

    ``serialized_chars`` measures the dict exactly as
    ``resources.node_to_dict`` ships it (provenance merged with
    ``source_traces`` and ``corrections``), with delivery char semantics.
    """

    full = {**provenance, "source_traces": source_traces, "corrections": corrections}
    return {
        "keys": {
            key: {
                "kind": type(value).__name__,
                **({"count": len(value)} if isinstance(value, (list, dict)) else {}),
                "chars": delivery_chars(value),
            }
            for key, value in provenance.items()
        },
        "serialized_chars": delivery_chars(full),
    }
