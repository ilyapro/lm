"""Production-shaped tests for the shared confirmatory-v4 seed validator."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from living_memory.config import (
    DEFAULT_RETRIEVAL_WEIGHTS,
    MemoryConfig,
    RetrievalWeightConfig,
)
from living_memory.storage import MemoryStore


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ap_confirmatory_probe_v4.py"


@pytest.fixture(scope="module")
def probe() -> Any:
    spec = importlib.util.spec_from_file_location("ap_confirmatory_probe_v4_seed_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _failure(probe: Any, call: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    with pytest.raises(probe.IntegrityFailure) as caught:
        call(*args, **kwargs)
    assert caught.value.args == ()
    assert str(caught.value) == ""


def _validate_path(probe: Any, path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        probe.validate_production_seed_state(connection)
    finally:
        connection.close()


def _fresh_store(path: Path, *, weights: dict[str, RetrievalWeightConfig] | None = None) -> None:
    config = MemoryConfig(
        db_path=path,
        retrieval_weights=(
            dict(DEFAULT_RETRIEVAL_WEIGHTS) if weights is None else dict(weights)
        ),
    )
    with MemoryStore(config):
        pass


def _populated_store(path: Path) -> tuple[str, str]:
    with MemoryStore(path) as store:
        left = store.append_trace("left", {"scope": "project:seed"})
        right = store.append_trace("right", {"scope": "project:seed"})
        store.create_connection(left.id, right.id, "related", weight=0.75)
        return left.id, right.id


def test_fresh_default_memorystore_is_valid_production_seed(probe: Any, tmp_path: Path) -> None:
    path = tmp_path / "fresh.sqlite3"
    _fresh_store(path)
    _validate_path(probe, path)


def test_unicode_concrete_learned_project_and_session_keys_validate(
    probe: Any, tmp_path: Path
) -> None:
    path = tmp_path / "learned.sqlite3"
    with MemoryStore(path) as store:
        store.set_retrieval_weights(
            "project:learned-π", bm25=0.6, vector=0.3, graph=0.1
        )
        store.set_retrieval_weights(
            "session:learned-σ", bm25=0.5, vector=0.25, graph=0.25
        )
    _validate_path(probe, path)


@pytest.mark.parametrize(
    ("key", "kind"),
    [
        ("default", "default"),
        ("project", "project"),
        ("global", "global"),
        ("session", "session"),
        ("project:p", "project"),
        ("project:a:b", "project"),
        ("session:s", "session"),
        ("session:λ", "session"),
    ],
)
def test_policy_key_mapper_accepts_only_explicit_supported_forms(
    probe: Any, key: str, kind: str
) -> None:
    assert probe.production_policy_key_kind(key) == kind


@pytest.mark.parametrize(
    "key",
    [
        "",
        "project:",
        "session:",
        "scope:project:p",
        "unknown",
        "global:x",
        "default:x",
        1,
        False,
    ],
)
def test_policy_key_mapper_rejects_unknown_empty_and_wrong_types(
    probe: Any, key: Any
) -> None:
    _failure(probe, probe.production_policy_key_kind, key)


def test_empty_production_table_is_not_seed_proof(probe: Any, tmp_path: Path) -> None:
    path = tmp_path / "empty.sqlite3"
    _fresh_store(path, weights={})
    connection = sqlite3.connect(path)
    try:
        _failure(probe, probe.validate_production_seed_state, connection)
    finally:
        connection.close()


@pytest.mark.parametrize("missing", ["default", "project", "global", "session"])
def test_each_required_policy_key_is_mandatory(
    probe: Any, tmp_path: Path, missing: str
) -> None:
    path = tmp_path / f"missing-{missing}.sqlite3"
    _fresh_store(path)
    # Mutate only after MemoryStore has closed; reopening it would silently
    # reseed the missing row and invalidate this test.
    connection = sqlite3.connect(path)
    try:
        connection.execute("DELETE FROM retrieval_weights WHERE scope = ?", (missing,))
        connection.commit()
        _failure(probe, probe.validate_production_seed_state, connection)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "unknown", ["unknown", "project:", "session:", "scope:project:p", "global:x"]
)
def test_unknown_policy_row_rejects_even_with_all_required_rows_present(
    probe: Any, tmp_path: Path, unknown: str
) -> None:
    path = tmp_path / "unknown.sqlite3"
    _fresh_store(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "INSERT INTO retrieval_weights "
            "(scope,bm25,vector,graph,learning_rate,updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (unknown, 0.4, 0.4, 0.2, 0.05, "2026-08-15T00:00:00Z"),
        )
        connection.commit()
        _failure(probe, probe.validate_production_seed_state, connection)
    finally:
        connection.close()


def test_duplicate_policy_key_rejects_without_primary_key_masking_it(
    probe: Any, tmp_path: Path
) -> None:
    path = tmp_path / "duplicate.sqlite3"
    _fresh_store(path)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            ALTER TABLE retrieval_weights RENAME TO retrieval_weights_original;
            CREATE TABLE retrieval_weights (scope, bm25, vector, graph);
            INSERT INTO retrieval_weights
                SELECT scope, bm25, vector, graph FROM retrieval_weights_original;
            INSERT INTO retrieval_weights VALUES ('default', 1.0, 0.0, 0.0);
            DROP TABLE retrieval_weights_original;
            """
        )
        connection.commit()
        _failure(probe, probe.validate_production_seed_state, connection)
    finally:
        connection.close()


@pytest.mark.parametrize("column", ["bm25", "vector", "graph"])
@pytest.mark.parametrize("bad", [float("inf"), "not-a-number", None])
def test_nonfinite_nonnumeric_or_null_weight_rejects(
    probe: Any, tmp_path: Path, column: str, bad: Any
) -> None:
    path = tmp_path / "bad-weight.sqlite3"
    _fresh_store(path)
    connection = sqlite3.connect(path)
    try:
        if bad is None:
            connection.executescript(
                """
                ALTER TABLE retrieval_weights RENAME TO retrieval_weights_original;
                CREATE TABLE retrieval_weights (scope, bm25, vector, graph);
                INSERT INTO retrieval_weights
                    SELECT scope, bm25, vector, graph FROM retrieval_weights_original;
                DROP TABLE retrieval_weights_original;
                """
            )
        connection.execute(
            f"UPDATE retrieval_weights SET {column} = ? WHERE scope = 'default'",
            (bad,),
        )
        connection.commit()
        _failure(probe, probe.validate_production_seed_state, connection)
    finally:
        connection.close()


def test_missing_projected_column_rejects(probe: Any, tmp_path: Path) -> None:
    path = tmp_path / "missing-column.sqlite3"
    _fresh_store(path)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            ALTER TABLE retrieval_weights RENAME TO retrieval_weights_original;
            CREATE TABLE retrieval_weights (scope, bm25, vector);
            INSERT INTO retrieval_weights
                SELECT scope, bm25, vector FROM retrieval_weights_original;
            DROP TABLE retrieval_weights_original;
            """
        )
        connection.commit()
        _failure(probe, probe.validate_production_seed_state, connection)
    finally:
        connection.close()


SEED_PROJECTION_COLUMNS = {
    "nodes": ("id", "level", "content", "scope", "created_at"),
    "connections": ("source_id", "target_id", "type", "weight"),
    "retrieval_weights": ("scope", "bm25", "vector", "graph"),
}


@pytest.mark.parametrize("table", tuple(SEED_PROJECTION_COLUMNS))
def test_each_projected_table_is_required(
    probe: Any, tmp_path: Path, table: str
) -> None:
    path = tmp_path / f"missing-table-{table}.sqlite3"
    _fresh_store(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute(f"DROP TABLE {table}")
        connection.commit()
        _failure(probe, probe.validate_production_seed_state, connection)
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("table", "missing"),
    [
        (table, column)
        for table, columns in SEED_PROJECTION_COLUMNS.items()
        for column in columns
    ],
)
def test_every_projected_column_is_required(
    probe: Any, tmp_path: Path, table: str, missing: str
) -> None:
    path = tmp_path / f"missing-{table}-{missing}.sqlite3"
    _fresh_store(path)
    kept = [column for column in SEED_PROJECTION_COLUMNS[table] if column != missing]
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            f"""
            ALTER TABLE {table} RENAME TO {table}_original;
            CREATE TABLE {table} ({','.join(kept)});
            DROP TABLE {table}_original;
            """
        )
        connection.commit()
        _failure(probe, probe.validate_production_seed_state, connection)
    finally:
        connection.close()


@pytest.mark.parametrize("table", tuple(SEED_PROJECTION_COLUMNS))
def test_seed_projection_views_are_rejected(
    probe: Any, tmp_path: Path, table: str
) -> None:
    path = tmp_path / f"view-{table}.sqlite3"
    _fresh_store(path)
    columns = SEED_PROJECTION_COLUMNS[table]
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            f"""
            ALTER TABLE {table} RENAME TO {table}_original;
            CREATE VIEW {table} AS
                SELECT {','.join(columns)} FROM {table}_original;
            """
        )
        connection.commit()
        _failure(probe, probe.validate_production_seed_state, connection)
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("column", "bad"),
    [
        ("id", ""),
        ("id", sqlite3.Binary(b"blob-id")),
        ("level", "unknown"),
        ("content", sqlite3.Binary(b"blob-content")),
        ("scope", "project:"),
        ("created_at", "not-utc"),
    ],
)
def test_node_field_types_and_grammars_are_strict(
    probe: Any, tmp_path: Path, column: str, bad: Any
) -> None:
    path = tmp_path / f"bad-node-{column}.sqlite3"
    left, _ = _populated_store(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.execute(
            f"UPDATE nodes SET {column} = ? WHERE id = ?", (bad, left)
        )
        connection.commit()
        _failure(probe, probe.validate_production_seed_state, connection)
    finally:
        connection.close()


def test_duplicate_node_and_connection_rows_are_rejected(
    probe: Any, tmp_path: Path
) -> None:
    node_path = tmp_path / "duplicate-node.sqlite3"
    left, _ = _populated_store(node_path)
    connection = sqlite3.connect(node_path)
    try:
        connection.executescript(
            """
            ALTER TABLE nodes RENAME TO nodes_original;
            CREATE TABLE nodes (id, level, content, scope, created_at);
            INSERT INTO nodes SELECT id,level,content,scope,created_at
                FROM nodes_original;
            """
        )
        connection.execute(
            "INSERT INTO nodes SELECT id,level,content,scope,created_at "
            "FROM nodes_original WHERE id = ?",
            (left,),
        )
        connection.commit()
        _failure(probe, probe.validate_production_seed_state, connection)
    finally:
        connection.close()

    edge_path = tmp_path / "duplicate-edge.sqlite3"
    _populated_store(edge_path)
    connection = sqlite3.connect(edge_path)
    try:
        connection.executescript(
            """
            ALTER TABLE connections RENAME TO connections_original;
            CREATE TABLE connections (source_id, target_id, type, weight);
            INSERT INTO connections SELECT source_id,target_id,type,weight
                FROM connections_original;
            INSERT INTO connections SELECT source_id,target_id,type,weight
                FROM connections_original;
            """
        )
        connection.commit()
        _failure(probe, probe.validate_production_seed_state, connection)
    finally:
        connection.close()


@pytest.mark.parametrize("mutation", ["target", "relation"])
def test_connection_target_and_relation_are_strict(
    probe: Any, tmp_path: Path, mutation: str
) -> None:
    path = tmp_path / f"bad-connection-{mutation}.sqlite3"
    _populated_store(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA ignore_check_constraints=ON")
        if mutation == "target":
            connection.execute("UPDATE connections SET target_id = 'missing'")
        else:
            connection.execute("UPDATE connections SET type = 'unknown'")
        connection.commit()
        _failure(probe, probe.validate_production_seed_state, connection)
    finally:
        connection.close()


def test_wrong_and_closed_connection_reject_silently(
    probe: Any, tmp_path: Path
) -> None:
    _failure(probe, probe.validate_production_seed_state, object())
    path = tmp_path / "closed.sqlite3"
    _fresh_store(path)
    connection = sqlite3.connect(path)
    connection.close()
    _failure(probe, probe.validate_production_seed_state, connection)


def test_temp_schema_cannot_shadow_empty_main_policy_table(
    probe: Any, tmp_path: Path
) -> None:
    path = tmp_path / "temp-shadow.sqlite3"
    _fresh_store(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute("DELETE FROM main.retrieval_weights")
        connection.execute(
            "CREATE TEMP TABLE retrieval_weights "
            "(scope, bm25, vector, graph)"
        )
        connection.executemany(
            "INSERT INTO temp.retrieval_weights VALUES (?,?,?,?)",
            [
                ("default", 1.0, 0.0, 0.0),
                ("project", 0.7, 0.3, 0.0),
                ("global", 0.4, 0.4, 0.2),
                ("session", 0.8, 0.2, 0.0),
            ],
        )
        connection.commit()
        _failure(probe, probe.validate_production_seed_state, connection)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "mutation",
    ["node-scope", "node-created-at", "dangling-edge", "bad-edge-weight"],
)
def test_node_and_connection_projection_failures_use_memorystore_origin(
    probe: Any, tmp_path: Path, mutation: str
) -> None:
    path = tmp_path / f"seed-{mutation}.sqlite3"
    left, _right = _populated_store(path)
    connection = sqlite3.connect(path)
    try:
        if mutation == "node-scope":
            connection.execute("UPDATE nodes SET scope = 'default' WHERE id = ?", (left,))
        elif mutation == "node-created-at":
            connection.execute("UPDATE nodes SET created_at = '+00:00' WHERE id = ?", (left,))
        elif mutation == "dangling-edge":
            connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute("UPDATE connections SET source_id = 'missing-node'")
        else:
            connection.execute("UPDATE connections SET weight = 'not-finite'")
        connection.commit()
        _failure(probe, probe.validate_production_seed_state, connection)
    finally:
        connection.close()


def test_policy_validation_is_insertion_order_invariant(probe: Any, tmp_path: Path) -> None:
    forward = {
        **DEFAULT_RETRIEVAL_WEIGHTS,
        "project:learned": RetrievalWeightConfig(0.5, 0.4, 0.1),
        "session:learned": RetrievalWeightConfig(0.6, 0.3, 0.1),
    }
    reverse = dict(reversed(list(forward.items())))
    first = tmp_path / "forward.sqlite3"
    second = tmp_path / "reverse.sqlite3"
    _fresh_store(first, weights=forward)
    _fresh_store(second, weights=reverse)
    _validate_path(probe, first)
    _validate_path(probe, second)


def test_probe_module_has_no_living_memory_runtime_dependency() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "from living_memory" not in source
    assert "import living_memory" not in source


# --------------------------------------------------------------------------
# v4 regression: a retrieval-policy key is not a memory scope
#
# The retired readiness scanner validated ``retrieval_weights.scope`` with the
# node-scope predicate.  Three of the four keys the serving configuration seeds
# into every real database fail that predicate, so the scanner rejected every
# production-shaped database and passed only an empty table.  The tests below
# pin both halves of the repair: the grammars stay separate, and the emptiness
# that masked the defect is itself rejected.
# --------------------------------------------------------------------------


def test_production_policy_keys_are_exactly_what_the_serving_config_seeds(
    probe: Any, tmp_path: Path
) -> None:
    """Anchor the contract to the real serving default, not to a copy of it."""

    assert set(DEFAULT_RETRIEVAL_WEIGHTS) == probe.REQUIRED_PRODUCTION_POLICY_KEYS
    path = tmp_path / "seeded.sqlite3"
    _fresh_store(path)
    connection = sqlite3.connect(path)
    try:
        seeded = {
            row[0] for row in connection.execute("SELECT scope FROM retrieval_weights")
        }
    finally:
        connection.close()
    assert seeded == {"default", "project", "global", "session"}


def test_node_scope_predicate_would_reject_the_real_production_keys(
    probe: Any,
) -> None:
    """Demonstrate the defect, so a future merge of the two grammars fails."""

    node_verdicts = {
        key: probe._node_scope_valid(key)
        for key in sorted(probe.REQUIRED_PRODUCTION_POLICY_KEYS)
    }
    assert node_verdicts == {
        "default": False,
        "global": True,
        "project": False,
        "session": False,
    }
    # The policy grammar accepts every one of them.
    for key in probe.REQUIRED_PRODUCTION_POLICY_KEYS:
        assert probe.production_policy_key_kind(key) == key
    # And the two grammars disagree in the other direction too: a concrete
    # node scope spelling that is not a policy key must still be rejected.
    assert probe._node_scope_valid("project:seed") is True
    assert probe.production_policy_key_kind("project:seed") == "project"
    assert probe._node_scope_valid("scope:project:p") is False
    _failure(probe, probe.production_policy_key_kind, "scope:project:p")


def test_seed_validator_does_not_call_the_node_scope_predicate_on_policy_keys(
    probe: Any, tmp_path: Path
) -> None:
    """A production database must validate even if the node predicate is fatal.

    Poisoning ``_node_scope_valid`` to reject everything leaves the policy
    projection unaffected only when the validator uses its own grammar.  If a
    future edit routed policy keys back through the node predicate, this test
    fails while every other seed test keeps passing.
    """

    path = tmp_path / "production.sqlite3"
    _fresh_store(path)
    connection = sqlite3.connect(path)
    try:
        connection.execute("DELETE FROM nodes")
        connection.execute("DELETE FROM connections")
        connection.commit()
    finally:
        connection.close()

    original = probe._node_scope_valid
    probe._node_scope_valid = lambda value: False
    try:
        _validate_path(probe, path)
    finally:
        probe._node_scope_valid = original


def test_emptiness_that_masked_the_defect_is_itself_rejected(
    probe: Any, tmp_path: Path
) -> None:
    """The retired scanner passed precisely because the table was empty."""

    path = tmp_path / "empty-policy.sqlite3"
    _fresh_store(path, weights={})
    connection = sqlite3.connect(path)
    try:
        assert (
            connection.execute("SELECT count(*) FROM retrieval_weights").fetchone()[0]
            == 0
        )
    finally:
        connection.close()
    _failure(probe, _validate_path, probe, path)


def test_frozen_plan_forbids_the_node_grammar_for_policy_keys(probe: Any) -> None:
    plan = json.loads(
        (
            REPO_ROOT
            / "artifacts"
            / "animal-planet"
            / "evaluation"
            / "confirmatory-holdout-v4"
            / "analysis-plan.json"
        ).read_text(encoding="utf-8")
    )
    seed = plan["replayability"]["shared_seed_state_validation"]
    assert seed["contract_id"] == probe.PRODUCTION_SEED_CONTRACT_ID
    assert seed["node_scope_grammar_applied_to_retrieval_weight_policy_key"] is False
    assert seed["retrieval_weights_table_must_be_nonempty"] is True
    assert seed["production_policy_keys_required"] == [
        "default",
        "project",
        "global",
        "session",
    ]
    # Every accept/reject vector the plan publishes must hold against the
    # implementation, including the four bare keys the node grammar rejects.
    for vector in seed["policy_key_vectors"]:
        if vector["accepted"]:
            assert probe.production_policy_key_kind(vector["input"]) == vector["kind"]
        else:
            _failure(probe, probe.production_policy_key_kind, vector["input"])
