"""A schema states a procedure its writers declared, never a history of cases.

Procedural consolidation groups traces by ``task_pattern``/``procedure_id``.
Sharing a group key only says the traces are about one topic: in the live
corpus nearly every such trace is a case record (a decision, a rejected
alternative, an execution report). Numbering those records as steps turned
histories of unrelated tasks into a binding instruction list. The only
structural evidence that a trace is a step is the writer's own declaration
(``step_order``/``step`` or ``step_description``/``step_content``).

The oracle below is fixed before the implementation: for each group it names
the exact ordered step texts a schema must carry (empty: no active schema),
and two negative controls — the old whole-trace concatenation and a schema
that lost its last conditional step — must fail it.

All contents are synthetic; no text or identifier of a real project is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from living_memory.consolidation import memory_consolidate, memory_teach
from living_memory.retrieval import memory_recall
from living_memory.storage import MemoryStore

SCOPE = "project:fixture"
AGENTS = ("agent-a", "agent-b", "agent-c", "agent-d")
# Records are written minutes apart just before the test runs, so the TTL sweep
# of a pass leaves them alone and a correction taught now lands in their era.
START = datetime.now(UTC) - timedelta(hours=2)


def _moment(minutes: int) -> str:
    return (START + timedelta(minutes=minutes)).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class Group:
    """One procedure group: what writers stored and what a schema must say."""

    name: str
    task_pattern: str
    procedure_id: str | None
    # (content, extra context) in write order.
    records: list[tuple[str, dict]]
    expected_steps: list[str]
    case_texts: list[str] = field(default_factory=list)
    # Only fully declared groups: the old mechanism already got these right.
    legacy_ok: bool = False


def _cases(prefix: str, count: int) -> list[tuple[str, dict]]:
    topics = (
        "payments service: chose a retry budget of three after the outage review",
        "search indexer: rejected a nightly rebuild because the index is append only",
        "mobile client: merged the hotfix without a feature flag, owner approved",
        "billing export: reverted the batch size change, memory spiked on the runner",
        "docs site: kept the old router, migration postponed to next quarter",
        "analytics job: split the query in two after a timeout on large tenants",
    )
    return [(f"{prefix} case {index + 1} — {topics[index % len(topics)]}", {}) for index in range(count)]


def dev_groups() -> list[Group]:
    ordered = [
        "Freeze writes on the primary before taking the snapshot",
        "Take the snapshot and record its checksum",
        "Restore the snapshot on the replica and compare checksums",
    ]
    conditional = [
        "Run the migration dry run against a staging copy",
        "Read the dry run report",
        "Apply the migration during the maintenance window",
        "If the dry run reported any warning, stop and escalate to the owner instead of applying; otherwise close the window",
    ]
    repeated = "Confirm the ticket owner before touching shared configuration"
    corrected = [
        "Export the ledger to csv",
        "Upload the csv to the legacy bucket",
        "Notify finance in the team channel",
    ]
    history = _cases("rollout", 6)
    return [
        Group(
            name="ordered",
            task_pattern="replica-restore",
            procedure_id="replica_restore",
            records=[(text, {"step_order": index + 1}) for index, text in enumerate(ordered)]
            + _cases("restore", 2),
            expected_steps=ordered,
            case_texts=[text for text, _ in _cases("restore", 2)],
        ),
        Group(
            name="conditional",
            task_pattern="schema-migration",
            procedure_id="schema_migration",
            # Written out of order: the declared position, not arrival, orders steps.
            records=[(conditional[index], {"step_order": index + 1}) for index in (2, 0, 3, 1)]
            + _cases("migration", 3),
            expected_steps=conditional,
            case_texts=[text for text, _ in _cases("migration", 3)],
        ),
        Group(
            name="history",
            task_pattern="release-supervision",
            procedure_id="release_supervision",
            records=history,
            expected_steps=[],
            case_texts=[text for text, _ in history],
        ),
        Group(
            # Rationale records tagged with the step they argue for (the
            # decision-log shape): one instruction, however many records.
            name="described",
            task_pattern="shared-config",
            procedure_id="shared_config",
            records=[
                (f"Rejected alternative {index + 1}: edit the file directly", {
                    "step_order": 1,
                    "step_description": repeated,
                })
                for index in range(3)
            ],
            expected_steps=[repeated],
            case_texts=[f"Rejected alternative {index + 1}: edit the file directly" for index in range(3)],
            legacy_ok=True,
        ),
        Group(
            # One instruction restated in other words: still one step.
            name="repeated",
            task_pattern="token-rotation",
            procedure_id="token_rotation",
            records=[
                (text, {"step_order": 1})
                for text in (
                    "Rotate the deploy token before it expires",
                    "Rotate the deploy token ahead of its expiry",
                    "Rotate the deploy token a week before it expires",
                )
            ],
            expected_steps=["Rotate the deploy token a week before it expires"],
        ),
        Group(
            name="corrected",
            task_pattern="ledger-export",
            procedure_id="ledger_export",
            records=[(text, {"step_order": index + 1}) for index, text in enumerate(corrected)],
            # Step 2 is taught below; step 3 is restated later at its own position.
            expected_steps=[
                corrected[0],
                "Upload the csv to the new archive bucket; the legacy bucket is read only",
                "Notify finance by ticket, the team channel is no longer used",
            ],
        ),
    ]


def shared_trigger_groups() -> list[Group]:
    steps = ["Tag the release candidate", "Publish the changelog", "Announce the release"]
    cases = _cases("release", 4)
    return [
        Group(
            name="trigger-steps",
            task_pattern="release-steps",
            procedure_id="release",
            records=[(text, {"step_order": index + 1}) for index, text in enumerate(steps)],
            expected_steps=steps,
        ),
        Group(
            name="trigger-cases",
            task_pattern="release-cases",
            procedure_id="release",
            records=cases,
            expected_steps=[],
            case_texts=[text for text, _ in cases],
        ),
    ]


def holdout_groups() -> list[Group]:
    """Held out: other spellings of declarations and other case shapes."""

    routing = [
        "STEP 1 — decide the role from the working directory",
        "STEP 2 — outside a goal worktree: hand the request to the supervisor and stop",
        "STEP 3 — inside the goal worktree: review the diff and post one summary",
    ]
    history = [
        ("Execution 2026-01-04: the job ran on the old runner and passed", {"lesson_kind": "completion"}),
        ("Trap: the cache key ignores the branch name", {"lesson_kind": "pitfall"}),
        ("Rejected alternative: split the job per tenant", {"lesson_kind": "planning"}),
        ("Execution 2026-02-11: second run, flaky network, retried once", {"lesson_kind": "completion"}),
    ]
    return [
        Group(
            name="holdout-routing",
            task_pattern="review-routing",
            procedure_id="review_request",
            # ``step`` as a string, one record written twice, decisions around it.
            records=[(text, {"step": str(index + 1)}) for index, text in enumerate(routing)]
            + [(routing[2], {"step": "3"})]
            + [("Review of ticket 7 went through the supervisor as expected", {})],
            expected_steps=routing,
            case_texts=["Review of ticket 7 went through the supervisor as expected"],
        ),
        Group(
            name="holdout-content-key",
            task_pattern="nightly-backup",
            procedure_id=None,
            records=[
                ("backup run notes", {"step_content": "Stop the writer service", "step": 1}),
                ("backup run notes, day two", {"step_content": "Copy the volume", "step": 2}),
                ("backup run notes, day three", {"step_content": "Start the writer service", "step": 3}),
            ],
            expected_steps=["Stop the writer service", "Copy the volume", "Start the writer service"],
            legacy_ok=True,
        ),
        Group(
            name="holdout-history",
            task_pattern="ci-job",
            procedure_id="ci_job",
            records=history,
            expected_steps=[],
            case_texts=[text for text, _ in history],
        ),
    ]


def _write(store: MemoryStore, group: Group) -> list:
    written = []
    for index, (content, extra) in enumerate(group.records):
        context = {
            "scope": SCOPE,
            "agent": AGENTS[index % len(AGENTS)],
            "task_pattern": group.task_pattern,
            "timestamp": _moment(index),
            **extra,
        }
        if group.procedure_id:
            context["procedure_id"] = group.procedure_id
        written.append(
            store.append_trace(content, context, feedback={"confidence": 0.6, "usefulness_score": 0.4})
        )
    return written


def _apply_corrections(store: MemoryStore, group: Group, written: list) -> None:
    if group.name != "corrected":
        return
    memory_teach(store, written[1].id, group.expected_steps[1], context={"agent": "agent-z"})
    store.append_trace(
        group.expected_steps[2],
        {
            "scope": SCOPE,
            "agent": "agent-z",
            "task_pattern": group.task_pattern,
            "procedure_id": group.procedure_id,
            "step_order": 3,
            "timestamp": _moment(90),
        },
    )


def _schemas_for(store: MemoryStore, task_pattern: str, *, include_decayed: bool = False) -> list:
    return [
        schema
        for schema in store.list_nodes(
            level="schema", scope=SCOPE, include_decayed=include_decayed, limit=100_000
        )
        if schema.context.get("task_pattern") == task_pattern
    ]


def oracle(group: Group, steps: list[str] | None) -> bool:
    """``steps`` is the active schema's procedure (None: no active schema)."""

    if not group.expected_steps:
        return steps is None
    return steps == group.expected_steps


def legacy_concatenation(store: MemoryStore, group: Group) -> list[str]:
    """What the replaced mechanism made of a group: every trace is a step."""

    traces = store.list_nodes_by_context(
        scope=SCOPE, context_filters={"task_pattern": group.task_pattern}, limit=1000
    )
    steps: list[str] = []
    for trace in sorted(traces, key=lambda node: (node.timestamp or "", node.id)):
        text = trace.context.get("step_description") or trace.context.get("step_content")
        step = str(text) if text else f"{trace.task}: {trace.content}" if trace.task else trace.content
        if step not in steps:
            steps.append(step)
    return steps


def _consolidated(tmp_path: Path, groups: list[Group]) -> tuple[MemoryStore, dict[str, list]]:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    written = {group.name: _write(store, group) for group in groups}
    for group in groups:
        _apply_corrections(store, group, written[group.name])
    memory_consolidate(store, scope=SCOPE)
    return store, written


def _active_steps(store: MemoryStore, group: Group) -> list[str] | None:
    schemas = _schemas_for(store, group.task_pattern)
    assert len(schemas) <= 1, f"{group.name}: {len(schemas)} active schemas for one group"
    return list(schemas[0].context["procedure"]) if schemas else None


@pytest.mark.parametrize("group_set", ["dev", "shared-trigger", "holdout"])
def test_schemas_carry_exactly_the_declared_procedure(tmp_path: Path, group_set: str) -> None:
    groups = {
        "dev": dev_groups,
        "shared-trigger": shared_trigger_groups,
        "holdout": holdout_groups,
    }[group_set]()
    store, written = _consolidated(tmp_path, groups)
    with store:
        for group in groups:
            steps = _active_steps(store, group)
            assert oracle(group, steps), f"{group.name}: {steps!r}"
            if steps:
                schema = _schemas_for(store, group.task_pattern)[0]
                for case in group.case_texts:
                    assert case not in schema.content
                assert schema.content.splitlines()[1:] == [
                    f"{index}. {step}" for index, step in enumerate(steps, start=1)
                ]
            # Every record keeps its identity and stays reachable by exact lookup.
            found = {
                node.id
                for node in store.list_nodes_by_context(
                    scope=SCOPE,
                    context_filters={"task_pattern": group.task_pattern},
                    include_decayed=True,
                    limit=1000,
                )
            }
            assert {trace.id for trace in written[group.name]} <= found


def test_negative_controls_fail_the_oracle(tmp_path: Path) -> None:
    groups = dev_groups() + holdout_groups()
    store, _written = _consolidated(tmp_path, groups)
    with store:
        for group in groups:
            if group.name == "corrected":
                continue  # the stand-in does not model era filtering
            legacy = legacy_concatenation(store, group)
            assert oracle(group, legacy) is group.legacy_ok, group.name
        conditional = next(group for group in groups if group.name == "conditional")
        truncated = _active_steps(store, conditional)[:-1]
        assert not oracle(conditional, truncated)
        assert not oracle(conditional, [step[:60] for step in conditional.expected_steps])


def test_existing_case_history_schema_is_retired_by_an_ordinary_pass(tmp_path: Path) -> None:
    """A schema the old mechanism built from cases is repaired without hand edits."""

    history = next(group for group in dev_groups() if group.name == "history")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        written = _write(store, history)
        legacy = legacy_concatenation(store, history)
        oversized = store.create_node(
            level="schema",
            content="\n".join(
                [f"Procedure: {history.procedure_id}"]
                + [f"{index}. {step}" for index, step in enumerate(legacy, start=1)]
            ),
            context={
                "scope": SCOPE,
                "agent": "memory_consolidate",
                "procedure_key": history.task_pattern,
                "task_pattern": history.task_pattern,
                "procedure_id": history.procedure_id,
                "trigger": "release supervision",
                "procedure": legacy,
            },
            provenance={
                "source_traces": sorted(trace.id for trace in written),
                "strategy": "procedural",
                "procedure_key": history.task_pattern,
            },
        )

        memory_consolidate(store, scope=SCOPE)
        memory_consolidate(store, scope=SCOPE)

        assert _schemas_for(store, history.task_pattern) == []
        retired = store.get_node(oversized.id)
        assert retired is not None and retired.decayed
        assert retired.content == oversized.content
        for trace in written:
            kept = store.get_node(trace.id)
            assert kept is not None and not kept.decayed and kept.content == trace.content

        results = memory_recall(store, "release supervision rollout case", scope=SCOPE, max_results=10)
        assert all(result.node.level != "schema" for result in results)
        assert {result.node.id for result in results} & {trace.id for trace in written}


def test_passes_new_records_and_teach_keep_one_schema_per_group(tmp_path: Path) -> None:
    groups = dev_groups() + shared_trigger_groups()
    store, written = _consolidated(tmp_path, groups)
    with store:
        before = {
            schema.id: schema.context["procedure"]
            for schema in store.list_nodes(level="schema", scope=SCOPE, limit=100_000)
        }
        second = memory_consolidate(store, scope=SCOPE)
        assert second.schemas_created == []
        after = {
            schema.id: schema.context["procedure"]
            for schema in store.list_nodes(level="schema", scope=SCOPE, limit=100_000)
        }
        assert after == before

        ordered = next(group for group in groups if group.name == "ordered")
        schema_id = _schemas_for(store, ordered.task_pattern)[0].id
        # A new case adds no step; a new declared step extends the same schema.
        _write(store, Group("ordered", ordered.task_pattern, ordered.procedure_id, _cases("late", 1), []))
        memory_consolidate(store, scope=SCOPE)
        assert _active_steps(store, ordered) == ordered.expected_steps
        store.append_trace(
            "Resume writes on the primary",
            {
                "scope": SCOPE,
                "agent": "agent-a",
                "task_pattern": ordered.task_pattern,
                "procedure_id": ordered.procedure_id,
                "step_order": 4,
            },
        )
        memory_consolidate(store, scope=SCOPE)
        assert _active_steps(store, ordered) == [*ordered.expected_steps, "Resume writes on the primary"]
        assert [schema.id for schema in _schemas_for(store, ordered.task_pattern)] == [schema_id]

        # Teaching a step that carried a step_description replaces its text.
        described = next(group for group in groups if group.name == "described")
        memory_teach(
            store,
            written["described"][2].id,
            "Confirm the ticket owner and the on-call before touching shared configuration",
        )
        memory_consolidate(store, scope=SCOPE)
        assert _active_steps(store, described) == [
            "Confirm the ticket owner and the on-call before touching shared configuration"
        ]
        total = store.list_nodes(level="schema", scope=SCOPE, limit=100_000)
        assert len(total) == len({schema.context["procedure_key"] for schema in total})


# A whole instruction often lives in one record with no step metadata at all:
# a conditional recipe, later corrected by teach. It is not a schema here, and
# it must not need to be: the oracle asks that ordinary consolidation, then a
# recall phrased the way a user asks, then the lookup a delivery points to,
# yield the current recipe complete — every condition and the last step.
RECIPE_PATTERN = "staging-cert-rotation"
RECIPE_V1 = (
    "Procedure: rotate the staging certificate. Use it when the staging certificate "
    "expires within 14 days. 1. Request a new certificate from the internal authority. "
    "2. If the certificate is a wildcard, install it on every ingress; otherwise install "
    "it on the gateway only. 3. Reload the ingress. 4. If the health check fails after "
    "the reload, restore the previous bundle and page the owner."
)
RECIPE_V2 = (
    "Procedure: rotate the staging certificate. Use it when the staging certificate "
    "expires within 14 days. 1. Request a new certificate from the internal authority. "
    "2. If the certificate is a wildcard, install it on the shared ingress only; otherwise "
    "install it on the gateway only. 3. Reload the ingress. 4. If the health check fails "
    "after the reload, restore the previous bundle, keep the new one for inspection and "
    "page the owner."
)
RECIPE_QUERY = "how do I rotate the staging certificate before it expires"


def _recipe_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from living_memory.server import create_mcp_server
    from test_transport_identity import FakeMCP

    monkeypatch.delenv("LM_AUTO_CONSOLIDATE_POLICY", raising=False)
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    context = {"scope": SCOPE, "task_pattern": RECIPE_PATTERN, "procedure_id": "staging_cert_rotation"}
    recipe = mcp.tools["memory_remember"](RECIPE_V1, {**context, "agent": "agent-a"})
    for index, (case, _extra) in enumerate(_cases("certificate", 4)):
        mcp.tools["memory_remember"](case, {**context, "agent": AGENTS[index]})
    mcp.tools["memory_teach"](recipe["node"]["id"], RECIPE_V2, context={"agent": "agent-z"})
    return mcp


def recipe_obtained(mcp) -> str | None:
    """The first recipe a user gets for ``RECIPE_QUERY``, re-fetched in full."""

    recalled = mcp.tools["memory_recall"](RECIPE_QUERY, scope=SCOPE, max_results=5)
    for result in recalled["results"]:
        node = result["node"]
        if not node["content"].startswith("Procedure: rotate the staging certificate"):
            continue
        if result.get("delivery", "full") == "full":
            return node["content"]
        ref = result["content_ref"]["node_id"]
        return mcp.tools["memory_lookup"](node_id=ref)["results"][0]["content"]
    return None


def test_unmarked_recipe_stays_whole_and_current_through_recall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp = _recipe_server(tmp_path, monkeypatch)
    mcp.tools["memory_consolidate"](scope=SCOPE, force=True)

    assert recipe_obtained(mcp) == RECIPE_V2
    store = mcp.memory_store
    assert _schemas_for(store, RECIPE_PATTERN) == []
    # The original survives as evidence of what was corrected.
    everything = store.list_nodes_by_context(
        scope=SCOPE, context_filters={"task_pattern": RECIPE_PATTERN}, include_decayed=True
    )
    assert RECIPE_V1 in {node.content for node in everything}


def test_negative_control_dropping_unmarked_instructions_fails_the_recipe_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp = _recipe_server(tmp_path, monkeypatch)
    store = mcp.memory_store
    for node in store.list_nodes(level="trace", scope=SCOPE, limit=1000):
        if not any(key in node.context for key in ("step", "step_order", "step_description", "step_content")):
            store.soft_delete_node(node.id, "negative control")
    mcp.tools["memory_consolidate"](scope=SCOPE, force=True)

    assert recipe_obtained(mcp) != RECIPE_V2
