"""Bounded canonical reuse preserves tokenizer consumers and policy changes."""

import pytest

from living_memory import embeddings as e


@pytest.fixture(autouse=True)
def reset_policy(monkeypatch):
    monkeypatch.setenv(e.CYRILLIC_STEM_ENV_VAR, "on")
    e.reset_cyrillic_stem_cache()
    yield
    e.reset_cyrillic_stem_cache()


def test_default_and_evidence_share_results_without_sharing_output_lists():
    text = "HTTP_429 если не готово users users базой ошибки"
    default = ["http", "готов", "user", "user", "database", "failure"]
    evidence = ["http", "429", "если", "не", "готов", "user", "user", "database", "failure"]
    first = e.tokenize(text)
    assert first == default
    assert e._canonical_token_cached.cache_info().misses == 5
    assert e.tokenize(text, evidence=True) == evidence
    assert e._canonical_token_cached.cache_info().misses == 8
    first.clear()
    assert e.tokenize(text) == default
    assert e.tokenize(text, evidence=True) == evidence
    assert e._canonical_token_cached.cache_info().misses == 8


@pytest.mark.parametrize("switch", ["on", "off"])
def test_reuse_preserves_tokens_and_default_embedding_features(monkeypatch, switch):
    monkeypatch.setenv(e.CYRILLIC_STEM_ENV_VAR, switch)
    e.reset_cyrillic_stem_cache()
    text = " ".join(e._SYNONYMS) + " базой пользователей HTTP_429 если не готово users users"
    tokens = [e.tokenize(text, evidence=mode) for mode in (False, True)]
    vector = e._embed_hash(text, 128)
    normalize = e._canonical_token_cached.__wrapped__
    with monkeypatch.context() as patch:
        patch.setattr(e, "_canonical_token", lambda token: normalize(token, e.cyrillic_stem_enabled()))
        assert [e.tokenize(text, evidence=mode) for mode in (False, True)] == tokens
        assert e._embed_hash(text, 128) == vector


def test_policy_reset_releases_results_and_applies_each_transition(monkeypatch):
    for switch, expected in [("on", "пользовател"), ("off", "пользователей"), ("on", "пользовател")]:
        monkeypatch.setenv(e.CYRILLIC_STEM_ENV_VAR, switch)
        e.reset_cyrillic_stem_cache()
        assert e._canonical_token_cached.cache_info().currsize == 0
        assert e.tokenize("пользователей пользователей") == [expected, expected]
        monkeypatch.setenv(e.CYRILLIC_STEM_ENV_VAR, "off" if switch == "on" else "on")
        assert e.tokenize("пользователей") == [expected]
        assert e._canonical_token_cached.cache_info().misses == 1


def test_cached_normalization_uses_only_its_explicit_policy(monkeypatch):
    # An in-flight call may finish after reset. Its policy stays in its key,
    # so it cannot supply the result of a later call under another policy.
    monkeypatch.setenv(e.CYRILLIC_STEM_ENV_VAR, "off")
    assert e._canonical_token_cached("базой", True) == "database"
    assert e._canonical_token_cached("базой", False) == "базой"
    monkeypatch.setenv(e.CYRILLIC_STEM_ENV_VAR, "on")
    e.reset_cyrillic_stem_cache()
    assert e._canonical_token_cached("базой", False) == "базой"
    assert e.tokenize("базой") == ["database"]


def test_entry_bound_evicts_without_changing_results():
    cache = e._canonical_token_cached
    limit = cache.cache_info().maxsize
    assert limit is not None
    for index in range(limit + 1):
        assert e.tokenize(f"item{index}") == [f"item{index}"]
    assert cache.cache_info().currsize == limit
    misses = cache.cache_info().misses
    assert e.tokenize("item0 item0") == ["item0", "item0"]
    assert cache.cache_info().misses == misses + 1
    assert cache.cache_info().currsize == limit


@pytest.mark.parametrize("switch", ["on", "off"])
def test_long_tokens_normalize_in_full_without_retention(monkeypatch, switch):
    monkeypatch.setenv(e.CYRILLIC_STEM_ENV_VAR, switch)
    cache = e._canonical_token_cached
    for token in ("x" * e._CANONICAL_TOKEN_MAX_LENGTH + "s", "пользователей" * 100, "4" * 1000):
        expected = cache.__wrapped__(token, switch == "on")
        assert e.tokenize(token, evidence=True) == [expected]
        assert e.tokenize(token, evidence=True) == [expected]
    assert cache.cache_info().currsize == 0
    assert cache.cache_info().misses == 0
