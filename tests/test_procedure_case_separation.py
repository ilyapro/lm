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

import re
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


def test_existing_case_history_schema_becomes_a_nonbinding_carrier_in_place(tmp_path: Path) -> None:
    """A schema the old mechanism built from cases is repaired without hand edits:
    the same node, feedback and edges stop binding and carry the cases as evidence.

    Replaces the retirement oracle of the first landing (node decayed with its
    numbered history): retiring it removed the only thing that carried some
    current records to the user."""

    history = next(group for group in dev_groups() if group.name == "history")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        written = _write(store, history)
        oversized = _legacy_schema(store, written, "release supervision", history.task_pattern)

        memory_consolidate(store, scope=SCOPE)
        first = store.get_node(oversized.id)
        memory_consolidate(store, scope=SCOPE)

        assert _schemas_for(store, history.task_pattern) == []
        assert_carrier_of(store, oversized.id, written)
        assert store.get_node(oversized.id).content == first.content
        for trace in written:
            kept = store.get_node(trace.id)
            assert kept is not None and not kept.decayed and kept.content == trace.content

        results = memory_recall(store, "release supervision rollout case", scope=SCOPE, max_results=10)
        assert all(result.node.level != "schema" for result in results)


def assert_carrier_of(store: MemoryStore, node_id: str, records: list) -> None:
    """``node_id`` is live, nonbinding, and holds exactly ``records``, whole and by id, no steps."""

    carrier = store.get_node(node_id)
    assert carrier is not None and not carrier.decayed and carrier.level == "concept"
    assert "procedure" not in carrier.context
    assert carrier.content.startswith("Evidence: ") and "not steps" in carrier.content
    assert not any(line[:1].isdigit() for line in carrier.content.splitlines())
    assert evidence_ids(carrier.content) == {record.id for record in records}
    assert all(store.get_node(record.id).content.strip() in carrier.content for record in records)


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


# Supervisor #1329 oracles, fixed before any further repair. A legacy
# case-history schema is often the only thing that carried a current recipe to
# the user: agents marked the recipe's own record irrelevant on unrelated
# questions, so a live valve keeps that record out of recall while the schema,
# bundled with unrelated cases, still reached the user. Retiring the schema
# must not be a new loss of the current recipe on the query that obtained it
# before, and an orphan schema (a group with no records left for a pass to
# visit) must be repaired by the same ordinary pass.
UNRELATED_QUESTIONS = (
    "which queue does the billing export use",
    "how is the search index compacted",
    "who approves a production schema migration",
)


EVIDENCE_LINE = re.compile(r"^- ([0-9A-Z]{26}): ", re.MULTILINE)


def evidence_ids(content: str) -> set[str]:
    return set(EVIDENCE_LINE.findall(content))


def obtained_texts(mcp, query: str, max_results: int = 5) -> list[tuple[str, str]]:
    """``(level, text)`` a reader obtains for ``query``: every delivery, re-fetched
    whole when cut. The policy reads only the query and what recall delivered:
    no target id, no expansion of the ids a carrier lists (it holds its records
    whole), nothing undelivered searched for."""

    obtained = []
    for result in mcp.tools["memory_recall"](query, scope=SCOPE, max_results=max_results)["results"]:
        node = result["node"]
        if result.get("delivery", "full") != "full":
            ref = (result.get("content_ref") or {}).get("node_id") or node["id"]
            node = mcp.tools["memory_lookup"](node_id=ref)["results"][0]
        obtained.append((node["level"], node["content"]))
    return obtained


def recipe_obtained_whole(mcp, query: str = RECIPE_QUERY, recipe: str = RECIPE_V2) -> str | None:
    """``query`` recalled naturally: the level of what brought the current
    recipe to the user complete (``None``: it never arrived)."""

    return next((level for level, text in obtained_texts(mcp, query) if recipe in text), None)


def _legacy_schema(store: MemoryStore, records: list, trigger: str, pattern: str):
    """The schema the replaced mechanism built: every record of a group numbered."""

    steps = [record.content for record in records]
    return store.create_node(
        level="schema",
        content="\n".join([f"Procedure: {trigger}"] + [f"{i}. {s}" for i, s in enumerate(steps, 1)]),
        context={
            "scope": SCOPE,
            "agent": "memory_consolidate",
            "procedure_key": pattern,
            "task_pattern": pattern,
            "trigger": trigger,
            "procedure": steps,
        },
        provenance={
            "source_traces": sorted(record.id for record in records),
            "strategy": "procedural",
            "procedure_key": pattern,
        },
    )


def _carried_recipe_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The recipe fixture as the live store holds it: a legacy schema carries the
    current recipe, whose own record the hub valve demotes."""

    monkeypatch.setenv("LM_HUB_SUPPRESSION_FACTOR", "0.1")
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "0.35")
    mcp = _recipe_server(tmp_path, monkeypatch)
    store = mcp.memory_store
    group = [
        node
        for node in store.list_nodes_by_context(scope=SCOPE, context_filters={"task_pattern": RECIPE_PATTERN})
        if node.level == "trace"
    ]
    current = next(node for node in group if node.content == RECIPE_V2)
    schema = _legacy_schema(store, group, "staging cert rotation", RECIPE_PATTERN)
    at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    for question in UNRELATED_QUESTIONS:
        event = store.record_recall_event(query=question, scope=SCOPE, requested_scope=SCOPE).id
        store.record_feedback_marks(
            [
                {
                    "recall_event_id": event,
                    "node_id": current.id,
                    "mark": "irrelevant",
                    "accepted": True,
                    "via_tool": "memory_recall",
                    "marked_at": at,
                }
            ]
        )
    return mcp, current, schema


def test_orphan_case_history_schema_becomes_a_carrier_in_an_ordinary_pass(tmp_path: Path) -> None:
    """A legacy schema whose group has no records left to visit is still repaired."""

    history = next(group for group in dev_groups() if group.name == "history")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        written = _write(store, history)
        oversized = _legacy_schema(store, written, "release supervision", history.task_pattern)
        # The group falls below the floor: a pass that only visits groups never
        # meets this schema again.
        for trace in written[2:]:
            store.soft_delete_node(trace.id, "fixture: group left without records")

        memory_consolidate(store, scope=SCOPE)

        assert _schemas_for(store, history.task_pattern) == []
        assert_carrier_of(store, oversized.id, written[:2])
        for trace in written[:2]:
            kept = store.get_node(trace.id)
            assert kept is not None and not kept.decayed and kept.content == trace.content
        # A second pass changes nothing.
        settled = store.get_node(oversized.id)
        memory_consolidate(store, scope=SCOPE)
        assert store.get_node(oversized.id).content == settled.content
        assert _schemas_for(store, history.task_pattern) == []


def test_orphan_does_not_adopt_another_groups_sources(tmp_path: Path) -> None:
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        records = [
            store.append_trace(text, {"scope": SCOPE, "task_pattern": "group-b", "procedure_id": "shared"})
            for text, _ in _cases("group b", 3)
        ]
        orphan = _legacy_schema(store, records, "shared", "group-a")
        memory_consolidate(store, scope=SCOPE)

        assert store.get_node(orphan.id).decayed
        carriers = [node for node in store.list_nodes(level="concept", scope=SCOPE)
                    if node.context.get("task_pattern") == "group-b"]
        assert len(carriers) == 1
        assert_carrier_of(store, carriers[0].id, records)


def test_ordinary_pass_retires_duplicate_group_nodes(tmp_path: Path) -> None:
    history = next(group for group in dev_groups() if group.name == "history")
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        written = _write(store, history)
        original = _legacy_schema(store, written, "release supervision", history.task_pattern)
        duplicate = store.create_node(
            level="schema", content=original.content + "\nAdditional legacy entry",
            context=original.context, provenance=original.provenance,
        )
        memory_consolidate(store, scope=SCOPE)

        live = [node for node in (store.get_node(original.id), store.get_node(duplicate.id)) if not node.decayed]
        assert len(live) == 1
        assert_carrier_of(store, live[0].id, written)
        memory_consolidate(store, scope=SCOPE)
        assert_carrier_of(store, live[0].id, written)
        assert sum(not store.get_node(node.id).decayed for node in (original, duplicate)) == 1


def test_carried_recipe_oracle_detects_a_mutant_that_drops_its_carrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The oracle is sensitive to carrier loss: with the legacy schema the
    current recipe is obtained; a mutant that only drops that schema loses it."""

    mcp, _current, schema = _carried_recipe_server(tmp_path, monkeypatch)
    assert recipe_obtained_whole(mcp)
    mcp.memory_store.soft_delete_node(schema.id, "mutant: carrier dropped")
    assert not recipe_obtained_whole(mcp)


def test_ordinary_pass_keeps_every_recipe_its_legacy_schema_carried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, _current, schema = _carried_recipe_server(tmp_path, monkeypatch)
    mcp.tools["memory_consolidate"](scope=SCOPE, force=True)
    assert _schemas_for(mcp.memory_store, RECIPE_PATTERN) == []
    assert mcp.memory_store.get_node(schema.id).level == "concept"
    assert recipe_obtained_whole(mcp) not in (None, "schema")


def test_carried_recipe_oracle_detects_a_pass_that_retires_the_carrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutant of the first landing: the ordinary pass retires the schema
    instead of turning it into a carrier. The recipe is lost."""

    import living_memory.consolidation as consolidation

    mcp, _current, schema = _carried_recipe_server(tmp_path, monkeypatch)
    monkeypatch.setattr(
        consolidation,
        "_create_or_update_schema",
        lambda store, scope, key, traces, existing, **_: (store.soft_delete_node(existing.id, "mutant"), False),
    )
    mcp.tools["memory_consolidate"](scope=SCOPE, force=True)
    assert mcp.memory_store.get_node(schema.id).decayed
    assert not recipe_obtained_whole(mcp)


# The accumulated state: the group's case-history schema is live and is the
# only thing the query matches that reaches the current recipe (a correction
# whose text never repeats the procedure's name). An ordinary pass must turn
# that node, in place, into the group's carrier.
NAMED_PATTERN = "ledger-stall-recovery"
NAMED_V1 = (
    "Procedure: when the nightly job stops advancing. 1. Pause the consumer. 2. Compare "
    "the last committed offset with the source. 3. If they differ, replay from the "
    "committed offset; otherwise restart the worker. 4. If the replay fails twice, "
    "restore yesterday's snapshot and page the owner."
)
NAMED_V2 = (
    "Procedure: when the nightly job stops advancing. 1. Pause the consumer. 2. Compare "
    "the last committed offset with the source. 3. If they differ, replay from the "
    "committed offset; otherwise restart the worker. 4. If the replay fails twice, "
    "restore yesterday's snapshot, keep the failed batch for inspection and page the owner."
)
NAMED_QUERY = "how do I run the ledger stall recovery"
# Records that share the query's words but not the procedure.
NAMED_DISTRACTORS = (
    "ledger service: the stall alert fired twice during the recovery drill",
    "ledger dashboard shows stall minutes per recovery window",
    "recovery drill notes: ledger replica caught up after the stall",
    "stall budget for the ledger is ten minutes before recovery starts",
    "the ledger recovery runbook review moved to next sprint after the stall",
    "ledger stall and recovery metrics were added to the weekly report",
)


def _named_recipe_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The accumulated state: the group's case-history schema is live, and the
    current recipe, corrected by teach, lives only in its own record."""

    from living_memory.server import create_mcp_server
    from test_transport_identity import FakeMCP

    monkeypatch.delenv("LM_AUTO_CONSOLIDATE_POLICY", raising=False)
    monkeypatch.setenv("LM_HUB_SUPPRESSION_FACTOR", "0.1")
    monkeypatch.setenv("LM_RECALL_MIN_SCORE", "0.35")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    remember = mcp.tools["memory_remember"]
    context = {"scope": SCOPE, "task_pattern": NAMED_PATTERN, "procedure_id": "ledger_stall_recovery"}
    recipe = remember(NAMED_V1, {**context, "agent": "agent-a"})
    for index, (case, _extra) in enumerate(_cases("ledger", 3)):
        remember(case, {**context, "agent": AGENTS[index]})
    for text in NAMED_DISTRACTORS:
        remember(text, {"scope": SCOPE, "agent": "agent-d"})
    mcp.tools["memory_teach"](recipe["node"]["id"], NAMED_V2, context={"agent": "agent-z"})
    store = mcp.memory_store
    group = [
        node
        for node in store.list_nodes_by_context(scope=SCOPE, context_filters={"task_pattern": NAMED_PATTERN})
        if node.level == "trace"
    ]
    schema = _legacy_schema(store, group, "ledger stall recovery", NAMED_PATTERN)
    return mcp, schema


def test_taught_recipe_arrives_whole_through_its_legacy_schema_turned_carrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mcp, schema = _named_recipe_server(tmp_path, monkeypatch)
    mcp.tools["memory_consolidate"](scope=SCOPE, force=True)
    mcp.tools["memory_consolidate"](scope=SCOPE, force=True)

    store = mcp.memory_store
    assert _schemas_for(store, NAMED_PATTERN) == []
    current = [
        node
        for node in store.list_nodes_by_context(scope=SCOPE, context_filters={"task_pattern": NAMED_PATTERN})
        if node.level == "trace" and node.content != NAMED_V1
    ]
    assert_carrier_of(store, schema.id, current)
    assert recipe_obtained_whole(mcp, NAMED_QUERY, NAMED_V2) not in (None, "schema")


def test_identity_oracle_detects_a_mutant_that_rebuilds_under_a_new_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pass that does not find the legacy node by its group emits a fresh
    carrier: the node's id, feedback and edges are left behind.

    Replaces the retired-state variant. That state was never created: the
    store only soft-deletes nodes (``decayed`` with a ``decay_reason``, no
    ``DELETE FROM nodes``), so any schema a first-landing pass retired would
    still carry ``procedural: no declared steps``; the complete read-only
    backups of both live stores taken after the revert hold no node with any
    ``procedural`` decay reason. Its revival path was removed with it."""

    import living_memory.consolidation as consolidation

    mcp, schema = _named_recipe_server(tmp_path, monkeypatch)
    monkeypatch.setattr(consolidation, "_schema_group_id", lambda node: "mutant: group not found")
    mcp.tools["memory_consolidate"](scope=SCOPE, force=True)

    with pytest.raises(AssertionError):
        assert_carrier_of(mcp.memory_store, schema.id, [])


# Two losses the frozen candidate showed on the regression replay, pinned.
def test_carrier_keeps_an_older_era_record_that_nothing_superseded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The era rule decides what a schema speaks for; it must not take a current
    record (only in an older era, not superseded) out of its group's evidence."""

    import living_memory.consolidation as consolidation

    history = next(group for group in dev_groups() if group.name == "history")
    assess = consolidation._assess_cluster_eras

    def older_era_first(store, members):
        era = assess(store, members)
        oldest = min(era.live, key=lambda node: (node.timestamp or "", node.id))
        era.live.remove(oldest)
        era.excluded[oldest.id] = "obsolete-era"
        return era

    monkeypatch.setattr(consolidation, "_assess_cluster_eras", older_era_first)
    with MemoryStore(tmp_path / "memory.sqlite3") as store:
        written = _write(store, history)
        memory_consolidate(store, scope=SCOPE)
        carriers = [
            node
            for node in store.list_nodes(level="concept", scope=SCOPE, limit=1000)
            if node.context.get("task_pattern") == history.task_pattern
        ]
        assert len(carriers) == 1
        assert_carrier_of(store, carriers[0].id, written)


LONG_RECIPE = (
    "Background: the staging certificate is issued by the internal authority for the shared "
    "ingress and the gateway; the previous rotations were done by hand from the release "
    "checklist, and two of them left the gateway on the expired bundle for a day because the "
    "reload was skipped. The checklist was retired after the audit and this record replaces it. "
    "Scope: staging only; production follows the change calendar. Owner: the platform on-call. "
    "Inputs: the authority request form, the ingress list and the health check dashboard. "
) + RECIPE_V2


def test_carrier_loss_mutant_holding_record_leads_loses_a_long_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The carrier is the only thing that brings the current recipe; a carrier
    that holds a 600-character lead of each record (the refuted experiment)
    forces a lookup no query-only reader takes, so the recipe never arrives."""

    import sys

    import living_memory.consolidation as consolidation

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "RECIPE_V2", LONG_RECIPE)
    assert len(LONG_RECIPE) > consolidation.DIGEST_LEAD_MAX_CHARS
    mcp, _current, _schema = _carried_recipe_server(tmp_path, monkeypatch)
    mcp.tools["memory_consolidate"](scope=SCOPE, force=True)
    assert recipe_obtained_whole(mcp, recipe=LONG_RECIPE) == "concept"

    mcp, _current, _schema = _carried_recipe_server(tmp_path / "mutant", monkeypatch)
    whole = consolidation._format_evidence_content

    def leads(store, trigger, records):
        return "\n".join(
            line if not line.startswith("- ") else line[:30] + consolidation._digest_lead(line[30:])
            for line in whole(store, trigger, records).split("\n")
        )

    monkeypatch.setattr(consolidation, "_format_evidence_content", leads)
    mcp.tools["memory_consolidate"](scope=SCOPE, force=True)
    assert recipe_obtained_whole(mcp, recipe=LONG_RECIPE) is None


def _crowded_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, groups: int = 10):
    """``groups`` case-history groups share one trigger (one procedure_id, many
    task_patterns), and a single recipe record of that procedure belongs to a
    group of its own, too small for a carrier."""

    from living_memory.server import create_mcp_server
    from test_transport_identity import FakeMCP

    monkeypatch.delenv("LM_AUTO_CONSOLIDATE_POLICY", raising=False)
    monkeypatch.setenv("LM_RECALL_SCHEMA_DEDUP", "1")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    for index in range(groups):
        for case, _extra in _cases(f"lesson {index}", 3):
            store.append_trace(
                case, {"scope": SCOPE, "agent": "agent-a", "procedure_id": "reopen_lesson", "task_pattern": f"goal-{index}"}
            )
    recipe = (
        "Procedure: reopen lesson. Write the lesson before the next attempt: 1. read the reopen "
        "comment, 2. name the failed approach, 3. store one preventive rule for the reopen lesson."
    )
    store.append_trace(recipe, {"scope": SCOPE, "agent": "agent-b", "procedure_id": "reopen_lesson", "task_pattern": "goal-lone"})
    memory_consolidate(store, scope=SCOPE)
    carriers = [node for node in store.list_nodes(level="concept", scope=SCOPE, limit=1000)
                if node.context.get("trigger") == "reopen lesson"]
    assert len(carriers) == groups
    return mcp, recipe


def test_same_trigger_carriers_do_not_crowd_out_a_record_recall_delivers_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cheap counterexample of the group-identity candidate's new loss: under
    the live dedup valve the carriers of one title share one slot, as the legacy
    schemas they replaced did, so the recipe's own record still arrives. The
    kept carrier holds its records whole.

    Replaces the oracle that required one slot per group: ten same-trigger
    carriers then took all ten slots and pushed the recipe out."""

    mcp, recipe = _crowded_server(tmp_path, monkeypatch)
    try:
        obtained = obtained_texts(mcp, "how to reopen lesson", max_results=10)
        assert ("trace", recipe) in obtained
        carriers = [text for level, text in obtained if level == "concept"]
        assert len(carriers) == 1 and carriers[0].count("\n- ") == 3
    finally:
        mcp.memory_store.close()


def test_compound_recipe_query_stays_ahead_of_source_backed_same_trigger_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp, recipe = _crowded_server(tmp_path, monkeypatch)
    try:
        obtained = obtained_texts(
            mcp, "how to reopen lesson and name the failed approach before the next attempt",
            max_results=4,
        )
        assert ("trace", recipe) in obtained
        assert sum(level == "concept" for level, _text in obtained) <= 1
    finally:
        mcp.memory_store.close()


def test_crowding_oracle_detects_the_group_identity_mutant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from living_memory import consolidation, schema_dedup

    title = schema_dedup.schema_title
    monkeypatch.setattr(
        schema_dedup,
        "schema_title",
        lambda node: title(node) and (node.scope, consolidation._schema_group_id(node)),
    )
    mcp, recipe = _crowded_server(tmp_path, monkeypatch)
    try:
        obtained = obtained_texts(mcp, "how to reopen lesson", max_results=10)
        assert ("trace", recipe) in obtained
        # With group identity, each historical carrier consumes a distinct
        # slot. The useful direct recipe must still remain reachable.
        carriers = [text for level, text in obtained if level == "concept"]
        assert len(carriers) == 9
        assert len(set(carriers)) == 9
    finally:
        mcp.memory_store.close()


def _relabel_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A legacy schema found by one label; most of its group's current records
    carry another procedure_id (one task_pattern, many labels)."""

    from living_memory.server import create_mcp_server
    from test_transport_identity import FakeMCP

    monkeypatch.delenv("LM_AUTO_CONSOLIDATE_POLICY", raising=False)
    monkeypatch.setenv("LM_RECALL_SCHEMA_DEDUP", "1")
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", mcp_factory=FakeMCP)
    store = mcp.memory_store
    labels = ["relay_drain_audit", "ops_sweep", "ops_sweep", "ops_sweep"]
    records = [
        store.append_trace(text, {"scope": SCOPE, "agent": "agent-a", "procedure_id": label, "task_pattern": "grp-relabel"})
        for (text, _extra), label in zip(_cases("sweep", 4), labels)
    ]
    for index in range(12):  # records that share the query's words, not the group
        store.append_trace(f"relay drain audit note {index}: the relay drained in {index + 2} minutes", {"scope": SCOPE})
    schema = _legacy_schema(store, records, "relay drain audit", "grp-relabel")
    return mcp, schema, records


def test_group_node_keeps_the_trigger_it_is_found_by(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression-replay loss: a pass relabeled a legacy schema turned carrier
    with its records' majority procedure_id, and the query that reached the
    group through its old trigger lost two current instructions."""

    mcp, schema, records = _relabel_server(tmp_path, monkeypatch)
    for _ in range(2):
        memory_consolidate(mcp.memory_store, scope=SCOPE)
    assert mcp.memory_store.get_node(schema.id).context["trigger"] == "relay drain audit"
    carriers = [text for level, text in obtained_texts(mcp, "how to relay drain audit", 10) if level == "concept"]
    assert any(all(record.content in text for record in records) for text in carriers)


def test_trigger_oracle_detects_a_pass_that_relabels_the_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import living_memory.consolidation as consolidation

    select = consolidation._select_group_procedure_key
    monkeypatch.setattr(consolidation, "_select_group_procedure_key", lambda traces, label=None: select(traces))
    mcp, schema, records = _relabel_server(tmp_path, monkeypatch)
    memory_consolidate(mcp.memory_store, scope=SCOPE)
    assert mcp.memory_store.get_node(schema.id).context["trigger"] == "ops sweep"
    carriers = [text for level, text in obtained_texts(mcp, "how to relay drain audit", 10) if level == "concept"]
    assert not any(all(record.content in text for record in records) for text in carriers)
