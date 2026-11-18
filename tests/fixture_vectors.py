"""Explicit space for tests that supply synthetic query and chunk coordinates."""

from living_memory.embeddings import EncodedVector

FIXTURE_SPACE = "fixture:cosine"


def fixture_vector(values):
    return EncodedVector(values, FIXTURE_SPACE)
