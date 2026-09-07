"""Cyrillic stemming in ``living_memory.embeddings.tokenize``.

Three claims, each pinned separately:

1. **Non-Cyrillic byte-identity.** ``tests/fixtures/tokenize_master_non_cyrillic.jsonl.gz``
   holds 3219 distinct goldset queries and 300 node contents without Cyrillic
   letters, tokenized by the *previous* tokenizer (``master`` at the branch
   point, generated with ``scripts/tokenize_master_fixture.py``). The current
   ``tokenize`` must reproduce every record exactly, in both switch states.
2. **Inflections collapse.** Russian forms of one lemma tokenize to one token;
   listed synonym forms keep their canonical token and unlisted inflections
   reach it through the stem; short words stay intact.
3. **BM25 path.** ``retrieval._expanded_query`` emits FTS5 prefix terms for
   Russian content words, ``storage._fts_query`` keeps them as ``"stem"*``
   and drops the star elsewhere, and an inflected query finds an inflected
   node in a real ``unicode61`` index. ``LM_TOKENIZE_CYRILLIC_STEM=off``
   restores the previous expression byte for byte.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from living_memory import embeddings
from living_memory.embeddings import (
    CYRILLIC_STEM_ENV_VAR,
    _CYRILLIC_STEM_SYNONYMS,
    _SYNONYMS,
    _is_russian_token,
    _stem_russian,
    cyrillic_prefix_terms,
    cyrillic_stem_enabled,
    reset_cyrillic_stem_cache,
    tokenize,
)
from living_memory.retrieval import _expanded_query
from living_memory.storage import MemoryStore, _fts_query

FIXTURE = Path(__file__).parent / "fixtures" / "tokenize_master_non_cyrillic.jsonl.gz"


@pytest.fixture
def stemming(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(CYRILLIC_STEM_ENV_VAR, "on")
    reset_cyrillic_stem_cache()
    yield
    reset_cyrillic_stem_cache()


@pytest.fixture
def no_stemming(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(CYRILLIC_STEM_ENV_VAR, "off")
    reset_cyrillic_stem_cache()
    yield
    reset_cyrillic_stem_cache()


def _fixture_records() -> list[dict[str, object]]:
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# 1. Non-Cyrillic byte-identity with the previous tokenizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("switch", ["on", "off"])
def test_non_cyrillic_fixture_is_reproduced_exactly(
    monkeypatch: pytest.MonkeyPatch, switch: str
) -> None:
    monkeypatch.setenv(CYRILLIC_STEM_ENV_VAR, switch)
    reset_cyrillic_stem_cache()
    try:
        records = _fixture_records()
        assert len(records) >= 3000
        kinds = {record["kind"] for record in records}
        assert kinds == {"goldset_query", "node_content"}
        mismatches = [
            (record["text"], record["tokens"], tokenize(str(record["text"])))
            for record in records
            if tokenize(str(record["text"])) != record["tokens"]
        ]
        assert mismatches == []
    finally:
        reset_cyrillic_stem_cache()


_IDENTIFIER_CASES = [
    ("src/living_memory/retrieval.py", ["src", "liv", "memory", "retrieval", "py"]),
    ("TraceInspectorPanel", ["trace", "inspector", "panel"]),
    ("camelCaseIdentifier", ["camel", "case", "identifier"]),
    ("HTTPServerError", ["httpserver", "failure"]),
    ("recall_map_builder", ["recall", "map", "builder"]),
    ("snake_case_name_v2", ["snake", "case", "name", "v2"]),
    ("EZ-13771", ["ez"]),
    ("06f6422abc", ["06f6422abc"]),
    ("a1b2c3d4e5f6a7b8", ["a1b2c3d4e5f6a7b8"]),
    ("node-01M1WBR5", ["node", "01m1wbr5"]),
    ("storybook dev --ci", ["storybook", "dev", "ci"]),
    ("LM_RECALL_CREDIT_POLICY=off", ["lm", "recall", "credit", "policy", "off"]),
    ("x86_64-linux-gnu", ["x86", "linux", "gnu"]),
    ("grounding.py:181", ["ground", "py"]),
    ("3.12.3", []),
    ("ID_007", ["id"]),
    ("Café résumé naïve", ["café", "résumé", "naïve"]),
    ("processes tested caching", ["process", "test", "cache"]),
    ("The quick brown fox jumps", ["quick", "brown", "fox", "jump"]),
    # Mixed-script identifiers: a token that mixes scripts is neither Latin
    # nor Russian and takes the untouched path in both switch states.
    ("abcдеф", ["abcдеф"]),
    ("тест123", ["тест123"]),
    ("recall_map_v2_абв1", ["recall", "map", "v2", "абв1"]),
]


@pytest.mark.parametrize("switch", ["on", "off"])
@pytest.mark.parametrize(("text", "expected"), _IDENTIFIER_CASES)
def test_identifiers_tokenize_as_before(
    monkeypatch: pytest.MonkeyPatch, switch: str, text: str, expected: list[str]
) -> None:
    monkeypatch.setenv(CYRILLIC_STEM_ENV_VAR, switch)
    reset_cyrillic_stem_cache()
    try:
        assert tokenize(text) == expected
    finally:
        reset_cyrillic_stem_cache()


def test_mixed_script_text_stems_only_the_russian_words(stemming: None) -> None:
    # Latin parts are byte-identical to the previous tokenizer; the Russian
    # word in the same text is stemmed like any other Russian word.
    assert tokenize("recall_map_кириллица") == ["recall", "map", "кириллиц"]
    assert tokenize("https://example.com/path/to/резюме") == [
        "http", "example", "com", "path", "резюм",
    ]


def test_mixed_script_text_is_unchanged_with_stemming_off(no_stemming: None) -> None:
    assert tokenize("recall_map_кириллица") == ["recall", "map", "кириллица"]
    assert tokenize("https://example.com/path/to/резюме") == [
        "http", "example", "com", "path", "резюме",
    ]


# ---------------------------------------------------------------------------
# 2. Russian inflections collapse onto one token
# ---------------------------------------------------------------------------

_LEMMAS = [
    # noun, listed synonym: every form reaches the canonical token
    (["миграция", "миграции", "миграцию", "миграцией", "миграциями"], "schema"),
    (["база", "базы", "базе", "базу", "базой", "базах", "базами"], "database"),
    (["ошибка", "ошибки", "ошибку", "ошибкой", "ошибками", "ошибках"], "failure"),
    (["кэш", "кэша", "кэше", "кэшем"], "cache"),
    (["тест", "теста", "тесты", "тестов", "тестами"], "test"),
    # noun, unlisted: one stem
    (["пользователь", "пользователя", "пользователи", "пользователей", "пользователями"], "пользовател"),
    (["индекс", "индекса", "индексы", "индексом", "индексах"], "индекс"),
    (["надежность", "надежности", "надежностью"], "надежн"),
    # adjective
    (["быстрый", "быстрая", "быстрое", "быстрые", "быстрого", "быстрыми"], "быстр"),
    # verb, reflexive, perfective gerund
    (["работает", "работать", "работал", "работала", "работали", "работают"], "работа"),
    (["сломалась", "сломался", "сломались"], "слома"),
    (["сделав", "сделавши"], "сдела"),
    # superlative
    (["новейший", "новейшая"], "нов"),
]


@pytest.mark.parametrize(("forms", "expected"), _LEMMAS)
def test_inflections_of_one_lemma_share_one_token(
    stemming: None, forms: list[str], expected: str
) -> None:
    tokens = {form: tokenize(form) for form in forms}
    assert all(value == [expected] for value in tokens.values()), tokens


def test_yo_is_folded_before_stemming(stemming: None) -> None:
    assert tokenize("Развёртывание упало") == ["deployment", "failure"]
    assert tokenize("всё") == tokenize("все")


@pytest.mark.parametrize("word", ["код", "бд", "да", "нет", "три", "мне", "ты", "их"])
def test_short_words_stay_intact(stemming: None, word: str) -> None:
    assert _stem_russian(word) == word


def test_stem_shorter_than_three_letters_keeps_the_surface_form(stemming: None) -> None:
    # "этого" would stem to "эт"; the guard returns the word unchanged.
    assert _stem_russian("этого") == "этого"
    assert _stem_russian("эти") == "эти"


def test_stem_is_always_a_prefix_of_the_word(stemming: None) -> None:
    words = [form for forms, _ in _LEMMAS for form in forms] + ["данных", "медленно", "требуется"]
    for word in words:
        stem = _stem_russian(word)
        assert word.startswith(stem), (word, stem)


def test_listed_synonym_forms_keep_their_canonical_token(stemming: None) -> None:
    cyrillic_keys = [key for key in _SYNONYMS if _is_russian_token(key)]
    assert len(cyrillic_keys) == 69
    for key in cyrillic_keys:
        assert tokenize(key) == [_SYNONYMS[key]], key


def test_derived_stem_synonyms_reach_unlisted_inflections(stemming: None) -> None:
    assert _CYRILLIC_STEM_SYNONYMS["баз"] == "database"
    assert _CYRILLIC_STEM_SYNONYMS["миграц"] == "schema"
    assert _CYRILLIC_STEM_SYNONYMS["кэш"] == "cache"
    # No derived stem is itself a listed key, and no two keys disagree.
    assert not set(_CYRILLIC_STEM_SYNONYMS) & set(_SYNONYMS)
    for stem, canonical in _CYRILLIC_STEM_SYNONYMS.items():
        assert canonical in set(_SYNONYMS.values()), (stem, canonical)
    # "потому" -> "пот" is skipped: "потом" ("later") shares that stem.
    assert "пот" not in _CYRILLIC_STEM_SYNONYMS
    assert tokenize("потом") == ["пот"]
    assert tokenize("потому") == ["cause"]


def test_stem_that_becomes_a_stop_word_is_dropped_like_any_stop_word(stemming: None) -> None:
    assert tokenize("какой какая такая") == []


# ---------------------------------------------------------------------------
# 5. Env switch
# ---------------------------------------------------------------------------


def test_switch_off_restores_surface_forms(no_stemming: None) -> None:
    assert not cyrillic_stem_enabled()
    assert tokenize("миграциями пользователей") == ["миграциями", "пользователей"]
    assert tokenize("миграции базы") == ["schema", "database"]  # listed forms still map
    assert tokenize("кэш") == ["кэш"]  # the derived map is off too
    assert cyrillic_prefix_terms("миграции базы") == []


@pytest.mark.parametrize("value", ["on", "1", "true", "", "anything"])
def test_switch_defaults_to_on(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(CYRILLIC_STEM_ENV_VAR, value)
    reset_cyrillic_stem_cache()
    try:
        assert cyrillic_stem_enabled()
    finally:
        reset_cyrillic_stem_cache()


def test_switch_is_cached_until_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CYRILLIC_STEM_ENV_VAR, "on")
    reset_cyrillic_stem_cache()
    try:
        assert tokenize("пользователей") == ["пользовател"]
        monkeypatch.setenv(CYRILLIC_STEM_ENV_VAR, "off")
        assert tokenize("пользователей") == ["пользовател"]  # still cached
        reset_cyrillic_stem_cache()
        assert tokenize("пользователей") == ["пользователей"]
    finally:
        reset_cyrillic_stem_cache()


def test_unset_variable_means_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CYRILLIC_STEM_ENV_VAR, raising=False)
    reset_cyrillic_stem_cache()
    try:
        assert embeddings.cyrillic_stem_enabled()
    finally:
        reset_cyrillic_stem_cache()


# ---------------------------------------------------------------------------
# 3. BM25 path: prefix terms against the unstemmed unicode61 index
# ---------------------------------------------------------------------------


def test_prefix_terms_cover_russian_content_words_only(stemming: None) -> None:
    terms = cyrillic_prefix_terms("Всё про миграцию базы данных и кэш: recallMap_v2 тесты для кода")
    assert terms == ["миграц*", "тест*"]
    # "всё"/"про"/"и"/"для": stop words or too short; "кэш": three letters,
    # never stemmed; "базы"/"данных"/"кода": stems "баз"/"дан"/"код" are under
    # the four-letter prefix floor, so those words keep matching exactly as
    # before. Latin words never become prefix terms. "этого" (stem would be
    # "эт") keeps its surface form through the guard and is emitted as its
    # own prefix, so a word the stemmer leaves alone still matches its longer
    # inflections.
    assert cyrillic_prefix_terms("recall map tests этого") == ["этого*"]
    assert cyrillic_prefix_terms("ревью целей мок") == []


def test_prefix_terms_keep_the_yo_spelling_too(stemming: None) -> None:
    assert cyrillic_prefix_terms("ёлочные игрушки") == ["елочн*", "ёлочн*", "игрушк*"]


def test_expanded_query_appends_prefix_terms_once(stemming: None) -> None:
    expanded = _expanded_query("миграции базы данных")
    assert expanded == "миграции базы данных schema database data миграц*"
    # A Cyrillic stem with a prefix term is not repeated as an exact term.
    expanded = _expanded_query("пользователей индекса")
    assert expanded == "пользователей индекса пользовател* индекс*"


def test_expanded_query_is_unchanged_with_stemming_off(no_stemming: None) -> None:
    assert _expanded_query("миграции базы данных") == "миграции базы данных schema database data"
    assert _expanded_query("пользователей индекса") == "пользователей индекса пользователей индекса"


def test_fts_query_emits_cyrillic_prefix_terms_and_drops_other_stars(stemming: None) -> None:
    assert (
        _fts_query("миграции schema миграц* баз*")
        == '"миграции" OR "schema" OR "миграц"* OR "баз"*'
    )
    assert _fts_query("foo* bar tests/*.py") == '"foo" OR "bar" OR "tests" OR "py"'
    assert _fts_query("ёлочн* елочн*") == '"ёлочн"* OR "елочн"*'


def test_fts_query_drops_every_star_with_stemming_off(no_stemming: None) -> None:
    assert _fts_query("миграции schema миграц* баз*") == '"миграции" OR "schema" OR "миграц" OR "баз"'
    assert _fts_query("foo* bar") == '"foo" OR "bar"'


def test_inflected_query_finds_inflected_node_through_bm25(stemming: None, tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        node = store.append_trace(
            "Ночью перенесли пользовательские индексы на новый сервер",
            {"scope": "project:alpha", "agent": "agent-ru"},
        )
        store.append_trace(
            "Настроили экспорт отчётов в формате CSV",
            {"scope": "project:alpha", "agent": "agent-ru"},
        )
        query = "пользовательский индекс сервера"
        exact = store.search_content(query, scope="project:alpha", limit=5)
        assert [found.id for found, _ in exact] == []  # no surface form in common
        expanded = store.search_content(_expanded_query(query), scope="project:alpha", limit=5)
        assert [found.id for found, _ in expanded] == [node.id]
