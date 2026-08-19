"""Document-frequency statistics over ``nodes_fts`` (fts5vocab, c-TF-IDF input).

Three contracts are under test here.

*Counts come from the index the triggers maintain.* ``term_document_frequencies``
reads the ``nodes_fts_vocab`` reader ('row' type) and ``fts_document_count`` the
``nodes_fts_docsize`` shadow table; both track node insert and hard delete
through the ``nodes_fts`` triggers with no sync step of their own, and both keep
counting soft-deleted nodes — soft deletion touches no FTS-synced column, which
is the documented caveat.

*Input folding matches the index by construction.* Terms are folded by the same
``unicode61`` tokenizer that indexed the content (case folding, Latin-only
diacritic removal), so any casing or accenting of a word counts the same
documents, Cyrillic survives untouched, and an input that folds to zero or
multiple tokens maps to 0 rather than to a guess.

*The migration is idempotent.* ``CREATE VIRTUAL TABLE IF NOT EXISTS`` runs on
every open, so reopening an existing database changes nothing, and a database
from before the vocab table existed gains it on reopen — pinned the same way
``tests/test_usage_attestation.py`` pins the v8 ledger's reopen behaviour, by
dropping the table out-of-band and reopening.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from living_memory.storage import MemoryStore


@pytest.fixture()
def store(tmp_path: Path):
    with MemoryStore(tmp_path / "fts-vocab.sqlite3") as opened:
        yield opened


def test_df_counts_and_document_total(store: MemoryStore) -> None:
    for content in ("alpha beta", "alpha gamma", "alpha beta delta"):
        store.create_node(level="trace", content=content)

    frequencies = store.term_document_frequencies(
        ["alpha", "beta", "gamma", "delta", "missing"]
    )
    assert frequencies == {
        "alpha": 3,
        "beta": 2,
        "gamma": 1,
        "delta": 1,
        "missing": 0,
    }
    assert store.fts_document_count() == 3


def test_folding_matches_the_index_tokenizer(store: MemoryStore) -> None:
    store.create_node(level="trace", content="Café память Recall")

    # unicode61 case-folds everywhere and strips diacritics from Latin
    # characters only, so every variant of a word counts the same document
    # and Cyrillic round-trips untouched.
    frequencies = store.term_document_frequencies(
        ["café", "CAFE", "cafe", "ПАМЯТЬ", "память", "Recall", "recall"]
    )
    assert frequencies == {term: 1 for term in frequencies}

    # Inputs that fold to zero or multiple tokens map to 0 by contract:
    # callers split compounds and phrases into single words first.
    degenerate = store.term_document_frequencies(
        ["recall_map", "alpha beta", "", "!!!", "   "]
    )
    assert degenerate == {term: 0 for term in degenerate}


def test_duplicate_inputs_collapse_and_empty_input_is_cheap(
    store: MemoryStore,
) -> None:
    store.create_node(level="trace", content="alpha")

    frequencies = store.term_document_frequencies(["Alpha", "alpha", "alpha"])
    assert frequencies == {"Alpha": 1, "alpha": 1}

    assert store.term_document_frequencies([]) == {}


def test_df_tracks_insert_hard_delete_and_ignores_soft_delete(
    store: MemoryStore,
) -> None:
    first = store.create_node(level="trace", content="alpha one")
    assert store.term_document_frequencies(["alpha"]) == {"alpha": 1}
    assert store.fts_document_count() == 1

    second = store.create_node(level="trace", content="alpha two")
    assert store.term_document_frequencies(["alpha"]) == {"alpha": 2}
    assert store.fts_document_count() == 2

    # A hard delete exercises the nodes_fts_delete trigger; the vocab reader
    # needs no sync step of its own because it reads the FTS index directly.
    with store.connection:
        store.connection.execute("DELETE FROM nodes WHERE id = ?", (first.id,))
    assert store.term_document_frequencies(["alpha"]) == {"alpha": 1}
    assert store.fts_document_count() == 1

    # Soft deletion updates only decayed/decay_reason, no FTS-synced column,
    # so the row stays indexed and both statistics keep counting it.
    store.delete_node(second.id, "test cleanup")
    assert store.term_document_frequencies(["alpha"]) == {"alpha": 1}
    assert store.fts_document_count() == 1


def test_document_count_agrees_with_the_fts_index(store: MemoryStore) -> None:
    # Guards the docsize shadow-table dependency: if nodes_fts ever stopped
    # maintaining docsize (columnsize=0) or the triggers drifted, the cheap
    # count would silently diverge from the index itself.
    for content in ("one alpha", "two beta", "three gamma"):
        store.create_node(level="trace", content=content)
    indexed = store.connection.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0]
    assert store.fts_document_count() == int(indexed) == 3


def test_reopen_is_idempotent_and_a_pre_vocab_db_gains_the_table(
    tmp_path: Path,
) -> None:
    db = tmp_path / "reopen.sqlite3"
    with MemoryStore(db) as opened:
        opened.create_node(level="trace", content="alpha beta")

    # Second open runs the same CREATE ... IF NOT EXISTS script: no error,
    # same counts, vocab still queryable.
    with MemoryStore(db) as reopened:
        assert reopened.term_document_frequencies(["alpha", "beta"]) == {
            "alpha": 1,
            "beta": 1,
        }
        assert reopened.fts_document_count() == 1

    # A database from before the vocab table existed is the same file minus
    # the table; dropping it out-of-band simulates one, and reopening with
    # the current build is the entire migration.
    raw = sqlite3.connect(db)
    try:
        raw.execute("DROP TABLE nodes_fts_vocab")
        raw.commit()
    finally:
        raw.close()

    with MemoryStore(db) as migrated:
        assert migrated.term_document_frequencies(["beta"]) == {"beta": 1}
        assert migrated.fts_document_count() == 1
