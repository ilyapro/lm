"""Multilingual embeddings and Unicode-aware tokenization for recall."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from hashlib import blake2b
import importlib.util
import logging
import math
import os
from pathlib import Path
import re
import sys
from typing import Any


DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSIONS = 384

_LOG = logging.getLogger(__name__)
_WARNED_FALLBACKS: set[str] = set()

_ENGLISH_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "so",
    "that",
    "the",
    "their",
    "then",
    "there",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

_RUSSIAN_STOP_WORDS = {
    "а",
    "без",
    "бы",
    "в",
    "во",
    "для",
    "до",
    "его",
    "ее",
    "если",
    "же",
    "за",
    "и",
    "из",
    "или",
    "им",
    "их",
    "к",
    "как",
    "ко",
    "ли",
    "на",
    "не",
    "но",
    "о",
    "об",
    "от",
    "по",
    "под",
    "при",
    "с",
    "со",
    "так",
    "то",
    "у",
    "уже",
    "что",
    "это",
    "этот",
    "эти",
    "эту",
}

_STOP_WORDS = _ENGLISH_STOP_WORDS | _RUSSIAN_STOP_WORDS

_SYNONYMS = {
    "auth": "authentication",
    "authenticate": "authentication",
    "authenticated": "authentication",
    "authenticating": "authentication",
    "authorization": "authentication",
    "authorize": "authentication",
    "signin": "login",
    "sign": "login",
    "logged": "login",
    "logging": "login",
    "db": "database",
    "postgres": "database",
    "postgresql": "database",
    "sqlite": "database",
    "mysql": "database",
    "migration": "schema",
    "migrations": "schema",
    "deploy": "deployment",
    "deployed": "deployment",
    "deploying": "deployment",
    "release": "deployment",
    "released": "deployment",
    "outage": "incident",
    "incident": "incident",
    "failure": "failure",
    "failed": "failure",
    "failing": "failure",
    "fail": "failure",
    "broken": "failure",
    "bug": "failure",
    "error": "failure",
    "exception": "failure",
    "fault": "failure",
    "problem": "failure",
    "issue": "failure",
    "rootcause": "cause",
    "reason": "cause",
    "because": "cause",
    "caused": "cause",
    "causing": "cause",
    "requires": "require",
    "required": "require",
    "needed": "require",
    "needs": "require",
    "dependency": "require",
    "depends": "require",
    "absent": "missing",
    "missing": "missing",
    "latency": "slow",
    "timeout": "slow",
    "timeouts": "slow",
    "slowdown": "slow",
    "cache": "cache",
    "cached": "cache",
    "caching": "cache",
    "config": "configuration",
    "cfg": "configuration",
    "setting": "configuration",
    "settings": "configuration",
    "env": "environment",
    "prod": "production",
    "production": "production",
    "staging": "staging",
    "test": "test",
    "tests": "test",
    "аутентификация": "authentication",
    "аутентификации": "authentication",
    "авторизация": "authentication",
    "авторизации": "authentication",
    "вход": "login",
    "входа": "login",
    "логин": "login",
    "база": "database",
    "базы": "database",
    "базе": "database",
    "базу": "database",
    "бд": "database",
    "данных": "data",
    "данные": "data",
    "данными": "data",
    "миграция": "schema",
    "миграции": "schema",
    "миграцию": "schema",
    "схема": "schema",
    "схемы": "schema",
    "развертывание": "deployment",
    "развертывания": "deployment",
    "развертывании": "deployment",
    "деплой": "deployment",
    "деплоя": "deployment",
    "релиз": "deployment",
    "релиза": "deployment",
    "ошибка": "failure",
    "ошибки": "failure",
    "ошибку": "failure",
    "ошибкой": "failure",
    "сбой": "failure",
    "сбоя": "failure",
    "сбоем": "failure",
    "упал": "failure",
    "упала": "failure",
    "упало": "failure",
    "падает": "failure",
    "падение": "failure",
    "проблема": "failure",
    "проблемы": "failure",
    "причина": "cause",
    "причины": "cause",
    "потому": "cause",
    "зависимость": "require",
    "зависимости": "require",
    "требует": "require",
    "нужен": "require",
    "нужна": "require",
    "нужно": "require",
    "отсутствует": "missing",
    "отсутствовала": "missing",
    "отсутствовал": "missing",
    "отсутствовало": "missing",
    "таймаут": "slow",
    "задержка": "slow",
    "медленно": "slow",
    "кеш": "cache",
    "кэша": "cache",
    "конфигурация": "configuration",
    "конфигурации": "configuration",
    "конфигурацию": "configuration",
    "настройка": "configuration",
    "настройки": "configuration",
    "окружение": "environment",
    "продакшен": "production",
    "тест": "test",
    "теста": "test",
    "тесты": "test",
}

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

# ---------------------------------------------------------------------------
# Russian light stemmer (Snowball-Russian endings) for Cyrillic tokens.
# ---------------------------------------------------------------------------
#
# ``_stem`` used to suffix-strip ASCII tokens only, so a Cyrillic word was
# matched purely as its surface form: "миграция" / "миграцию" / "миграций"
# were three unrelated tokens for IDF containment, schema-trigger overlap and
# the BM25 query expansion. The stemmer below folds the inflections of one
# lemma onto one token. It is the Snowball Russian algorithm (perfective
# gerund, adjectival = adjective [+ participle], reflexive, verb, noun endings
# searched in RV; trailing "и"; derivational "-ость/-ост" in R2; superlative
# "-ейш/-айш"-style endings, "нн" undoubling and a trailing "ь"), with two
# guards that keep short words intact: tokens of three characters or fewer
# are never stemmed (the same floor the ASCII branch has) and a stem shorter
# than ``_RU_MIN_STEM`` characters is rejected in favour of the surface form.
#
# Only tokens made entirely of Russian letters are eligible. Anything else --
# Latin, digits, mixed-script identifiers, other Cyrillic alphabets -- takes
# exactly the path it took before, which is what keeps every non-Cyrillic
# input byte-identical to the previous tokenizer (pinned by
# tests/test_tokenize_cyrillic.py against a fixture generated from it).
#
# The switch ``LM_TOKENIZE_CYRILLIC_STEM`` (``on`` by default; ``off``, ``0``,
# ``false``, ``no`` or ``disabled`` restore the surface-form behaviour) is
# read once per process and cached; ``reset_cyrillic_stem_cache`` re-reads it
# so tests and benchmarks can flip it.
#
# The BM25 index (``nodes_fts``) is FTS5 ``unicode61`` and does not stem, so a
# stem such as "миграц" is never an index token. ``cyrillic_prefix_terms``
# turns the Russian content words of a query into FTS5 prefix terms
# (``миграц*``) that ``retrieval._expanded_query`` appends and
# ``storage._fts_query`` emits as ``"миграц"*``; the index itself is untouched.

CYRILLIC_STEM_ENV_VAR = "LM_TOKENIZE_CYRILLIC_STEM"
_CYRILLIC_STEM_OFF_VALUES = frozenset({"off", "0", "false", "no", "disabled"})
_CYRILLIC_STEM_ENABLED: bool | None = None

_RU_VOWELS = frozenset("аеиоуыэюя")
_RU_LETTERS = frozenset("абвгдежзийклмнопрстуфхцчшщъыьэюяё")
_RU_MIN_STEM = 3
#: Shortest stem that may become an FTS5 prefix term. A three-letter Cyrillic
#: prefix ("рев*" from "ревью", "цел*" from "цели", "дан*" from "данных")
#: covers a wide slice of the vocabulary and, measured on the retrieval
#: harness, pushed relevant nodes down more often than it lifted them; four
#: letters keeps the prefix specific to the lemma. A word whose stem is
#: shorter keeps matching exactly as before (its surface form stays in the
#: raw query).
_RU_MIN_PREFIX = 4


def _ru_endings(*groups: tuple[tuple[str, ...], bool]) -> tuple[tuple[str, bool], ...]:
    """Endings as ``(suffix, needs_a_or_ya)`` pairs, longest first.

    Snowball's ``among`` takes the longest suffix that fits inside the region
    and then applies that suffix's condition; a failed condition fails the
    whole step rather than falling back to a shorter suffix. Sorting once here
    lets ``_ru_longest`` reproduce that with a single scan.
    """

    flat = [(suffix, guarded) for suffixes, guarded in groups for suffix in suffixes]
    return tuple(sorted(flat, key=lambda item: -len(item[0])))


_RU_PERFECTIVE_GERUND = _ru_endings(
    (("в", "вши", "вшись"), True),
    (("ив", "ивши", "ившись", "ыв", "ывши", "ывшись"), False),
)
_RU_ADJECTIVE = _ru_endings(
    (
        (
            "ее", "ие", "ые", "ое", "ими", "ыми", "ей", "ий", "ый", "ой", "ем", "им",
            "ым", "ом", "его", "ого", "ему", "ому", "их", "ых", "ую", "юю", "ая", "яя",
            "ою", "ею",
        ),
        False,
    ),
)
_RU_PARTICIPLE = _ru_endings(
    (("ем", "нн", "вш", "ющ", "щ"), True),
    (("ивш", "ывш", "ующ"), False),
)
_RU_REFLEXIVE = _ru_endings((("ся", "сь"), False))
_RU_VERB = _ru_endings(
    (
        (
            "ла", "на", "ете", "йте", "ли", "й", "л", "ем", "н", "ло", "но", "ет", "ют",
            "ны", "ть", "ешь", "нно",
        ),
        True,
    ),
    (
        (
            "ила", "ыла", "ена", "ейте", "уйте", "ите", "или", "ыли", "ей", "уй", "ил",
            "ыл", "им", "ым", "ен", "ило", "ыло", "ено", "ят", "ует", "уют", "ит", "ыт",
            "ены", "ить", "ыть", "ишь", "ую", "ю",
        ),
        False,
    ),
)
_RU_NOUN = _ru_endings(
    (
        (
            "а", "ев", "ов", "ие", "ье", "е", "иями", "ями", "ами", "еи", "ии", "и", "ией",
            "ей", "ой", "ий", "й", "иям", "ям", "ием", "ем", "ам", "ом", "о", "у", "ах",
            "иях", "ях", "ы", "ь", "ию", "ью", "ю", "ия", "ья", "я",
        ),
        False,
    ),
)
_RU_SUPERLATIVE = _ru_endings((("ейш", "ейше", "айш", "айше"), False))
_RU_DERIVATIONAL = _ru_endings((("ост", "ость"), False))


def cyrillic_stem_enabled() -> bool:
    """Whether Cyrillic tokens are stemmed (``LM_TOKENIZE_CYRILLIC_STEM``)."""

    global _CYRILLIC_STEM_ENABLED
    if _CYRILLIC_STEM_ENABLED is None:
        raw = os.environ.get(CYRILLIC_STEM_ENV_VAR, "on").strip().lower()
        _CYRILLIC_STEM_ENABLED = raw not in _CYRILLIC_STEM_OFF_VALUES
    return _CYRILLIC_STEM_ENABLED


def reset_cyrillic_stem_cache() -> None:
    """Forget the cached ``LM_TOKENIZE_CYRILLIC_STEM`` reading."""

    global _CYRILLIC_STEM_ENABLED
    _CYRILLIC_STEM_ENABLED = None


def _is_russian_token(token: str) -> bool:
    return bool(token) and all(char in _RU_LETTERS for char in token)


def _ru_regions(word: str) -> tuple[int, int]:
    """``(RV, R2)`` start offsets in the Snowball sense.

    RV is the region after the first vowel; R1 the region after the first
    non-vowel that follows a vowel; R2 the same construction inside R1. R1 is
    only needed to derive R2. A region that does not exist starts at the end
    of the word, so nothing can be matched inside it.
    """

    length = len(word)
    rv = length
    for index, char in enumerate(word):
        if char in _RU_VOWELS:
            rv = index + 1
            break
    r1 = length
    for index in range(1, length):
        if word[index] not in _RU_VOWELS and word[index - 1] in _RU_VOWELS:
            r1 = index + 1
            break
    r2 = length
    for index in range(r1 + 1, length):
        if word[index] not in _RU_VOWELS and word[index - 1] in _RU_VOWELS:
            r2 = index + 1
            break
    return rv, r2


def _ru_longest(word: str, endings: tuple[tuple[str, bool], ...], region: int) -> tuple[str, bool] | None:
    """Longest ending that lies entirely inside ``word[region:]``."""

    for suffix, guarded in endings:
        if word.endswith(suffix) and len(word) - len(suffix) >= region:
            return suffix, guarded
    return None


def _ru_strip(word: str, endings: tuple[tuple[str, bool], ...], region: int) -> str | None:
    """Remove the longest fitting ending; ``None`` when the step fails.

    A guarded ending (Snowball's group 1) must be preceded by "а" or "я" and
    that letter must itself lie inside the region; otherwise the step fails
    without trying a shorter ending, exactly like the ``among`` it mirrors.
    """

    found = _ru_longest(word, endings, region)
    if found is None:
        return None
    suffix, guarded = found
    cut = len(word) - len(suffix)
    if guarded and (cut - 1 < region or word[cut - 1] not in ("а", "я")):
        return None
    return word[:cut]


def _stem_russian(token: str) -> str:
    """Snowball Russian stem of a lowercase, all-Cyrillic token."""

    word = token.replace("ё", "е")
    rv, r2 = _ru_regions(word)
    if rv >= len(word):
        return token

    stripped = _ru_strip(word, _RU_PERFECTIVE_GERUND, rv)
    if stripped is not None:
        word = stripped
    else:
        stripped = _ru_strip(word, _RU_REFLEXIVE, rv)
        if stripped is not None:
            word = stripped
        stripped = _ru_strip(word, _RU_ADJECTIVE, rv)
        if stripped is not None:
            word = stripped
            stripped = _ru_strip(word, _RU_PARTICIPLE, rv)
            if stripped is not None:
                word = stripped
        else:
            stripped = _ru_strip(word, _RU_VERB, rv)
            if stripped is None:
                stripped = _ru_strip(word, _RU_NOUN, rv)
            if stripped is not None:
                word = stripped

    if word.endswith("и") and len(word) - 1 >= rv:
        word = word[:-1]

    stripped = _ru_strip(word, _RU_DERIVATIONAL, r2)
    if stripped is not None:
        word = stripped

    if word.endswith("нн") and len(word) - 1 >= rv:
        word = word[:-1]
    else:
        stripped = _ru_strip(word, _RU_SUPERLATIVE, rv)
        if stripped is not None:
            word = stripped
            if word.endswith("нн") and len(word) - 1 >= rv:
                word = word[:-1]
        elif word.endswith("ь") and len(word) - 1 >= rv:
            word = word[:-1]

    if len(word) < _RU_MIN_STEM:
        return token
    return word


def _build_cyrillic_stem_synonyms() -> dict[str, str]:
    """Stem -> canonical token for every Cyrillic ``_SYNONYMS`` key.

    A direct ``_SYNONYMS`` hit still wins (``_canonical_token`` looks the
    surface form up first), so this only adds the inflections the table does
    not list: "базой" stems to "баз", the stem of the listed "база"/"базы"/
    "базе"/"базу", and reaches "database" through this map. A stem two keys
    disagree about is left unmapped, and keys in ``_CYRILLIC_STEM_SYNONYM_SKIP``
    are excluded because their stem is a common unrelated word.
    """

    derived: dict[str, str] = {}
    conflicts: set[str] = set()
    for key, value in _SYNONYMS.items():
        if key in _CYRILLIC_STEM_SYNONYM_SKIP or not _is_russian_token(key):
            continue
        stem = _stem_russian(key) if len(key) > 3 else key
        if stem in _SYNONYMS:
            continue
        previous = derived.get(stem)
        if previous is not None and previous != value:
            conflicts.add(stem)
        derived.setdefault(stem, value)
    for stem in conflicts:
        derived.pop(stem, None)
    return derived


#: Cyrillic synonym keys whose stem is a common unrelated word: "потому" stems
#: to "пот", which is also the stem of "потом" ("later"), so a derived
#: "пот" -> "cause" entry would relabel every "потом".
_CYRILLIC_STEM_SYNONYM_SKIP = frozenset({"потому"})
_CYRILLIC_STEM_SYNONYMS = _build_cyrillic_stem_synonyms()


def cyrillic_prefix_terms(text: str) -> list[str]:
    """FTS5 prefix terms (``стем*``) for the Russian content words of ``text``.

    One term per distinct stem, in order of first appearance, for every
    all-Cyrillic word of more than three letters that is not a stop word and
    whose stem has at least ``_RU_MIN_PREFIX`` (four) letters. A word spelled with
    "ё" yields a second term in the original spelling, because ``unicode61``
    keeps "ё" and "е" distinct while the stemmer folds them. Empty when
    Cyrillic stemming is off, which leaves the BM25 query exactly as before.
    """

    if not cyrillic_stem_enabled():
        return []
    separated = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    separated = separated.replace("_", " ").replace("-", " ").replace("/", " ").replace(".", " ")
    terms: dict[str, None] = {}
    for raw in _TOKEN_RE.findall(separated.lower()):
        folded = raw.replace("ё", "е")
        if len(folded) <= 3 or folded in _STOP_WORDS or not _is_russian_token(folded):
            continue
        stem = _stem_russian(folded)
        if len(stem) < _RU_MIN_PREFIX:
            continue
        terms.setdefault(f"{stem}*")
        if "ё" in raw:
            spelled = raw[: len(stem)]
            if spelled != stem:
                terms.setdefault(f"{spelled}*")
    return list(terms)


def fts_prefix_term_allowed(term: str) -> bool:
    """Whether ``storage._fts_query`` may emit ``term`` as an FTS5 prefix term."""

    return cyrillic_stem_enabled() and _is_russian_token(term)


_HASH_BACKENDS = {"hash", "fallback", "local-hash"}
_ONLINE_BACKENDS = {"online", "remote", "download", "network"}
_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")

_MODEL_CACHE: dict[tuple[str, str], Any] = {}


class LocalEmbeddingModel:
    """Lazy local sentence-transformers encoder with a deterministic fallback.

    The public class name is kept for compatibility with existing callers. It
    loads a configured sentence-transformers model only when it resolves to a
    local path or existing local cache. Missing dependencies, uncached model
    names, or load failures fall back to a Unicode-aware hashed encoder without
    network access.
    """

    def __init__(
        self,
        dimensions: int | None = None,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        backend: str | None = None,
    ) -> None:
        if dimensions is not None and dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = int(dimensions or EMBEDDING_DIMENSIONS)
        self.model_name = model_name
        configured_backend = backend or os.environ.get("LIVING_MEMORY_EMBEDDING_BACKEND") or "auto"
        self._backend = configured_backend.lower()
        self._model: Any | None = None
        self._load_attempted = False

    def embed(self, text: str) -> list[float]:
        model = self._sentence_transformer()
        if model is None:
            return _embed_hash(text, self.dimensions)

        try:
            vector = _encode_sentence_transformer(model, text)
        except Exception as exc:  # pragma: no cover - depends on optional backend behavior
            self._warn_fallback("sentence-transformers encode failed", exc)
            self._model = None
            self._backend = "hash"
            return _embed_hash(text, self.dimensions)

        if vector:
            self.dimensions = len(vector)
            return _normalize(vector)
        return [0.0] * self.dimensions

    def warmup(self) -> None:
        """Pre-load the underlying model to avoid latency on the first query."""
        self._sentence_transformer()

    def encoder_id(self) -> str:
        """Name the encoder ``embed`` uses right now, for keying cached vectors.

        The model name alone is not enough: an unavailable model silently
        falls back to the hashed encoder, whose vectors mean something else.
        """

        if self._sentence_transformer() is None:
            return f"hash:{self.dimensions}"
        return f"st:{self.model_name}:{self.dimensions}"

    def _sentence_transformer(self) -> Any | None:
        if self._backend in _HASH_BACKENDS:
            return None
            
        cache_key = (self.model_name, self._backend)
        if cache_key in _MODEL_CACHE:
            self._model = _MODEL_CACHE[cache_key]
            self._load_attempted = True
            dimension = self._model.get_sentence_embedding_dimension()
            if dimension:
                self.dimensions = int(dimension)
            return self._model

        if self._load_attempted:
            return self._model

        self._load_attempted = True
        model_source = self.model_name
        enforce_offline = self._backend not in _ONLINE_BACKENDS
        if enforce_offline:
            local_source = _resolve_local_model_source(self.model_name)
            if local_source is None:
                if _sentence_transformers_installed():
                    _LOG.debug(
                        "embedding model %r is not present locally; using hash-based fallback",
                        self.model_name,
                    )
                else:
                    self._warn_fallback("sentence-transformers is not installed")
                return None
            model_source = local_source

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            self._warn_fallback("sentence-transformers is not installed", exc)
            return None

        try:
            if enforce_offline:
                with _offline_model_loading():
                    self._model = SentenceTransformer(model_source)
            else:
                self._model = SentenceTransformer(model_source)
            dimension = self._model.get_sentence_embedding_dimension()
            if dimension:
                self.dimensions = int(dimension)
            _MODEL_CACHE[cache_key] = self._model
        except Exception as exc:  # pragma: no cover - depends on local model cache/network
            self._warn_fallback(f"could not load embedding model {model_source!r}", exc)
            self._model = None
        return self._model

    def _warn_fallback(self, reason: str, exc: BaseException | None = None) -> None:
        key = f"{self.model_name}:{reason}"
        if key in _WARNED_FALLBACKS:
            return
        _WARNED_FALLBACKS.add(key)
        if exc is None:
            _LOG.warning("%s; using hash-based multilingual fallback embeddings", reason)
            return
        _LOG.warning(
            "%s; using hash-based multilingual fallback embeddings",
            reason,
            exc_info=(type(exc), exc, exc.__traceback__),
        )


def _sentence_transformers_installed() -> bool:
    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except (ImportError, ValueError):
        return "sentence_transformers" in sys.modules


def _resolve_local_model_source(model_name: str) -> str | None:
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path)
    if _looks_like_path(model_name):
        return None

    return _cached_sentence_transformer_path(model_name) or _cached_huggingface_snapshot(model_name)


def _looks_like_path(model_name: str) -> bool:
    if model_name.startswith(("~", ".")):
        return True
    if os.path.isabs(model_name):
        return True
    if os.altsep is not None and os.altsep in model_name:
        return True
    if os.sep in model_name:
        return len([part for part in model_name.split(os.sep) if part]) > 2
    return False


def _cached_sentence_transformer_path(model_name: str) -> str | None:
    candidates: list[Path] = []
    cache_root = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
    cache_roots = [Path(cache_root).expanduser()] if cache_root else []
    cache_roots.append(Path.home() / ".cache" / "torch" / "sentence_transformers")

    names = {
        model_name,
        model_name.replace("/", "_"),
        model_name.split("/")[-1],
    }
    if "/" not in model_name:
        names.add(f"sentence-transformers_{model_name}")

    for root in cache_roots:
        for name in names:
            candidates.append(root / name)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _cached_huggingface_snapshot(model_name: str) -> str | None:
    for repo_id in _candidate_repo_ids(model_name):
        for hub_cache in _huggingface_cache_roots():
            model_cache = hub_cache / f"models--{repo_id.replace('/', '--')}"
            snapshot = _snapshot_from_refs(model_cache) or _newest_snapshot(model_cache)
            if snapshot is not None:
                return str(snapshot)
    return None


def _candidate_repo_ids(model_name: str) -> list[str]:
    repo_ids = [model_name]
    if "/" not in model_name:
        repo_ids.append(f"sentence-transformers/{model_name}")
    return repo_ids


def _huggingface_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for env_var in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(env_var)
        if value:
            roots.append(Path(value).expanduser())

    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    roots.append(hf_home / "hub")

    unique_roots: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root not in seen:
            unique_roots.append(root)
            seen.add(root)
    return unique_roots


def _snapshot_from_refs(model_cache: Path) -> Path | None:
    refs_dir = model_cache / "refs"
    for ref_name in ("main", "master"):
        ref_file = refs_dir / ref_name
        if not ref_file.exists():
            continue
        revision = ref_file.read_text(encoding="utf-8").strip()
        snapshot = model_cache / "snapshots" / revision
        if snapshot.exists():
            return snapshot
    return None


def _newest_snapshot(model_cache: Path) -> Path | None:
    snapshots_dir = model_cache / "snapshots"
    if not snapshots_dir.exists():
        return None

    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not snapshots:
        return None
    return max(snapshots, key=lambda path: path.stat().st_mtime)


@contextmanager
def _offline_model_loading() -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in _OFFLINE_ENV_VARS}
    for name in _OFFLINE_ENV_VARS:
        os.environ[name] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def tokenize(text: str) -> list[str]:
    """Return normalized Unicode tokens for BM25 expansion and fallback vectors."""

    separated = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    separated = separated.replace("_", " ").replace("-", " ").replace("/", " ").replace(".", " ")
    normalized = separated.lower().replace("ё", "е")
    raw_tokens = _TOKEN_RE.findall(normalized)

    tokens: list[str] = []
    for raw_token in raw_tokens:
        token = raw_token.strip("_")
        if not token or token.isdigit() or token in _STOP_WORDS:
            continue
        canonical = _canonical_token(token)
        if canonical and canonical not in _STOP_WORDS:
            tokens.append(canonical)
    return tokens


def cosine_similarity(left: Iterable[float] | None, right: Iterable[float] | None) -> float:
    """Compute cosine similarity for two embedding vectors."""

    if left is None or right is None:
        return 0.0

    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right, strict=False):
        left_float = float(left_value)
        right_float = float(right_value)
        dot += left_float * right_float
        left_norm += left_float * left_float
        right_norm += right_float * right_float

    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def _encode_sentence_transformer(model: Any, text: str) -> list[float]:
    try:
        vector = model.encode(text, normalize_embeddings=True)
    except TypeError:
        vector = model.encode(text)
    values = vector.tolist() if hasattr(vector, "tolist") else vector
    if values and isinstance(values[0], list):
        values = values[0]
    return [float(value) for value in values]


def _embed_hash(text: str, dimensions: int) -> list[float]:
    tokens = tokenize(text)
    if not tokens:
        return [0.0] * dimensions

    vector = [0.0] * dimensions
    for token in tokens:
        _add_feature(vector, f"tok:{token}", 1.0)
        stem = _stem(token)
        if stem != token:
            _add_feature(vector, f"stem:{stem}", 0.55)

    for left, right in zip(tokens, tokens[1:]):
        _add_feature(vector, f"bi:{left}:{right}", 0.6)

    compact = "".join(tokens)
    if len(compact) >= 4:
        for size in (3, 4):
            for index in range(max(0, len(compact) - size + 1)):
                _add_feature(vector, f"char:{compact[index:index + size]}", 0.08)

    return _normalize(vector)


def _canonical_token(token: str) -> str:
    direct = _SYNONYMS.get(token)
    if direct is not None:
        return direct
    stemmed = _stem(token)
    canonical = _SYNONYMS.get(stemmed)
    if canonical is None and cyrillic_stem_enabled():
        canonical = _CYRILLIC_STEM_SYNONYMS.get(stemmed)
    return stemmed if canonical is None else canonical


def _stem(token: str) -> str:
    if len(token) <= 3:
        return token
    if _is_latin_token(token):
        for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)]
        return token
    if cyrillic_stem_enabled() and _is_russian_token(token):
        return _stem_russian(token)
    return token


def _is_latin_token(token: str) -> bool:
    return all(("a" <= char <= "z") or char.isdigit() for char in token)


def _add_feature(vector: list[float], feature: str, weight: float) -> None:
    digest = blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    bucket = value % len(vector)
    sign = 1.0 if (value >> 63) == 0 else -1.0
    vector[bucket] += sign * weight


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return vector
    return [value / norm for value in vector]
