"""Grounded-usage attestation: the client submits evidence, the server decides.

``memory_attest`` is the write path that lets a *finished* session's recall
events earn the credit an in-session ``memory_remember`` would have given them.
Everything that makes it honest rather than merely convenient is pinned here,
and each rule below is one a client would otherwise be able to bend:

* **The server recomputes.** A payload that asserts ``grounded: true`` and
  ``containment: 1.0`` for every result must produce byte-identical output to
  the same payload without those keys. If it ever does not, the metric is
  self-reported and worthless.
* **Realistic evidence discriminates.** The negative control submits genuine
  diff and command text — the scale a real session produces, not a two-line
  toy — against nodes from another domain, and must earn exactly nothing. A
  threshold that only separates toys separates nothing in the field.
* **Per evidence item, never concatenated.** Five fragments that each fall well
  under the gate must stay ungrounded even though their concatenation clears it
  comfortably. This is the whole reason grounding runs per item: 0.25 was
  calibrated on trace-sized documents, and a long enough document contains
  every common token by accident.
* **Credit mirrors the live path.** Rank decay, the grounded subset only, the
  event's scope — the same arithmetic, reused rather than reimplemented.
* **The ledger, not ``feedback_applied``, is the replay guard.** Re-attesting
  the same ``(event, evidence)`` moves nothing at all.
* **Closure is a separate decision from credit**, and closing twice must never
  credit ``linked_count`` twice or clobber an existing trace pointer.

The seeded fixture is the one the credit and anchor work left behind — eight
delivered results sharing only the two-token query — so the numbers here are
comparable with ``tests/test_grounded_credit_assignment.py`` and
``tests/test_anchor_live_path.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import sqlite3

import pytest

from living_memory.attestation import (
    ATTESTATION_MIN_CONTAINMENT,
    EVIDENCE_MAX_ITEM_CHARS,
    EVIDENCE_MAX_ITEMS,
    EVIDENCE_MAX_TOTAL_CHARS,
    AttestationError,
    attest_recall_usage,
    canonical_evidence,
)
from living_memory.feedback import RECALL_CREDIT_MIN_CONTAINMENT
from living_memory.query_anchors import ANCHOR_EDGE_WEIGHT
from living_memory.server import create_mcp_server
from living_memory.storage import MemoryStore, RECALL_ATTESTATION_TABLE

SCOPE = "project:attest"
QUERY = "kappa ledger"
AMBIENT = {"agent": "seeder", "task": "attest-fixture", "session_id": "live-session"}
#: The offline extractor never shares the recalling session's identity — that is
#: the point of the feature, and nothing here may gate on it.
EXTRACTOR_CONTEXT = {
    "agent": "extractor",
    "task": "post-session-extraction",
    "session_id": "extraction-run-1",
    "source_session_key": "claude/2026-08-19T11-02-33Z/ab12cd",
}

#: Eight delivered results sharing only the two-token query, so evidence that
#: reuses one node's distinctive vocabulary grounds that node and nothing else.
BODIES = [
    "alembic migration checksum drift blocked the staging rollout entirely",
    "toolbar palette swatches moved into the ColorDock component last week",
    "cert-manager wildcard certificate renewal needs a dns01 solver token",
    "redis eviction storm traced to a runaway zset with unbounded members",
    "grafana dashboard panel queries broke after the datasource uid rename",
    "kafka consumer lag alert fires when the rebalance protocol thrashes",
    "terraform state lock stuck behind an abandoned dynamodb lease record",
    "webpack chunk splitting regressed the vendor bundle size by a third",
]
REDIS = 3
ALEMBIC = 0
CERT_MANAGER = 2


# ---------------------------------------------------------------------------
# Evidence fixtures: real artifact text, at the size a session actually emits
# ---------------------------------------------------------------------------

#: Every fixture below is a whole artifact that fits inside
#: ``EVIDENCE_MAX_ITEM_CHARS`` — which is 600, one trace-sized document, since
#: the field check re-fitted the caps (``docs/post-session-attestation.md``).
#: That is the size a *graded unit* has, not the size a session's whole diff
#: has: the assembler splits a longer hunk into cap-sized parts on line
#: boundaries, so nothing is discarded. These stay single items so the
#: structural assertions below (``evidence_items``, ``evidence_index``) keep
#: saying what they mean.

#: A diff that *uses* the redis note without quoting it: the fix it describes is
#: the one the recalled node warned about. Grounds node 3 at ~0.76.
EVICTION_DIFF = """\
diff --git a/services/cache.py b/services/cache.py
@@ -41,10 +41,17 @@ class SessionCache:
     def _touch(self, key: str) -> None:
-        self._redis.zadd(self._zset, {key: time.time()})
+        # The eviction storm was an unbounded zset: members were only ever
+        # added, never trimmed, so redis kept the runaway set resident
+        # until maxmemory-policy began dropping live keys.
+        pipeline = self._redis.pipeline()
+        pipeline.zadd(self._zset, {key: time.time()})
+        pipeline.zremrangebyrank(self._zset, 0, -(self.MAX_MEMBERS + 1))
+        pipeline.execute()
"""

#: Command output that uses the alembic note. Grounds node 0 at ~0.76.
ALEMBIC_OUTPUT = """\
$ alembic upgrade head
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
ERROR [alembic.util.messaging] Can't locate revision identified by '9c41ab77e0d2'
FAILED: Can't locate revision identified by '9c41ab77e0d2'

Same checksum drift that blocked the staging rollout: the branch rebased its
migration file after the revision had already been stamped, so the
down_revision recorded in staging no longer exists anywhere in the tree.
Re-stamped with `alembic stamp 4d7f0b19aa31` and the upgrade completed.
"""

#: Negative control. A real session's artifacts from an unrelated frontend task:
#: a component diff, its test run, and a lint log — three full graded units of
#: the same kind of text the positive fixtures are made of, sharing no subject
#: matter with any seeded node.
UNRELATED_DIFF = """\
diff --git a/web/src/InvoiceTable.tsx b/web/src/InvoiceTable.tsx
@@ -18,9 +18,18 @@ interface InvoiceRow {
   amountCents: number
+  settledAt: string | null
 }

-export function InvoiceTable({ rows }: Props) {
-  return <table>{rows.map(renderRow)}</table>
+export function InvoiceTable({ rows, locale }: Props) {
+  const sorted = useMemo(
+    () => [...rows].sort((a, b) => b.issuedAt.localeCompare(a.issuedAt)),
+    [rows],
+  )
+  return (
+    <table>{sorted.map((r) => <tr key={r.id} data-settled={!!r.settledAt} />)}</table>
+  )
 }
"""

UNRELATED_TEST_OUTPUT = """\
$ npm run test -- --runInBand web/src/components
> storefront@4.2.0 test
> jest --runInBand web/src/components

 PASS  web/src/components/InvoiceTable.test.tsx
  InvoiceTable
    renders one row per invoice (28 ms)
    sorts newest first (7 ms)
    formats amounts in the requested locale (5 ms)
    marks settled invoices with a data attribute (4 ms)

Test Suites: 1 passed, 1 total
Tests:       4 passed, 4 total
Snapshots:   0 total
Time:        2.417 s
"""

#: The third control artifact was originally a ``vite build`` log. It had to be
#: replaced: a log printing chunk and vendor bundle sizes grounded node 7
#: ("webpack chunk splitting regressed the vendor bundle size") at 0.28, which
#: is the measure working, not failing — that log really is about the same
#: subject. A control has to be evidence from another *domain*, not merely from
#: another repository.
UNRELATED_LINT_LOG = """\
$ npx eslint web/src --max-warnings 0
/repo/web/src/components/InvoiceTable.tsx
  24:9  warning  React Hook useMemo has a missing dependency: 'locale'
  41:5  warning  Prefer optional chaining over an explicit null comparison

/repo/web/src/routes/invoices/index.lazy.tsx
  12:1  error  'formatIssuedAt' is defined but never used  no-unused-vars

2 warnings and 1 error

$ git commit -am 'invoices: sort newest first and localise amounts'
[feature/invoice-sorting f81a204] invoices: sort newest first
 3 files changed, 55 insertions(+), 6 deletions(-)
"""

UNRELATED_EVIDENCE = [UNRELATED_DIFF, UNRELATED_TEST_OUTPUT, UNRELATED_LINT_LOG]

#: Five artifact fragments from one cert-manager session. Each carries a couple
#: of node 2's distinctive tokens and none clears the gate alone (max ~0.12);
#: their concatenation reaches ~0.63. Grading one joined document would credit
#: a node no single artifact evidences.
FRAGMENTED_EVIDENCE = [
    """\
$ kubectl -n ingress get pods
NAME                          READY   STATUS    RESTARTS   AGE
cert-manager-7f4b8d9c6-2xk4t  1/1     Running   0          6d
webhook-5c9d7f8b4-9qzlm       1/1     Running   0          6d""",
    """\
diff --git a/deploy/base/issuer.yaml b/deploy/base/issuer.yaml
@@ -8,6 +8,9 @@ spec:
   acme:
     server: https://acme-v02.api.letsencrypt.org/directory
+    solvers:
+      - selector: {}""",
    """\
$ openssl x509 -noout -text -in /tmp/edge.pem | head -3
Certificate:
    Data:
        Version: 3 (0x2)""",
    """\
2026-08-19T09:14:02Z  WARN  renewal deferred: waiting for propagation
2026-08-19T09:19:11Z  INFO  order valid, issuing wildcard chain for *.edge.example.com""",
    """\
$ vault read -field=token secret/acme/dns01
hvs.CAESIJ8redacted
$ echo $?
0""",
]


class FakeMCP:
    def __init__(self, name: str, instructions: str | None = None) -> None:
        self.name = name
        self.instructions = instructions
        self.tools: dict[str, Any] = {}
        self.resources: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}

    def tool(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.tools[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        return decorate if func is None else decorate(func)

    def resource(self, uri: str, **_kwargs: Any) -> Any:
        def decorate(func: Any) -> Any:
            self.resources[uri] = func
            return func

        return decorate

    def prompt(self, func: Any | None = None, **kwargs: Any) -> Any:
        def decorate(inner: Any) -> Any:
            self.prompts[str(kwargs.get("name") or inner.__name__)] = inner
            return inner

        return decorate if func is None else decorate(func)


@pytest.fixture(autouse=True)
def _hash_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    monkeypatch.delenv("LM_RETRIEVAL_TUNING_POLICY", raising=False)
    monkeypatch.delenv("LM_RECALL_CREDIT_POLICY", raising=False)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _seed(db_path: Path, order: list[int] | None = None) -> tuple[Any, MemoryStore, list[str], str]:
    """Eight seeded nodes plus one recall event delivering all eight.

    The event is recorded directly rather than through ``memory_recall`` so the
    *rank* of each result — the input to credit decay — is chosen by the test
    instead of by the ranker. ``order`` lists node indices in delivery order.
    """

    mcp = create_mcp_server(db_path, mcp_factory=FakeMCP, expose_attest=True)
    store: MemoryStore = mcp.memory_store
    node_ids = [
        store.append_trace(f"{QUERY} {body}", {"scope": SCOPE, **AMBIENT}).id
        for body in BODIES
    ]
    delivery = order if order is not None else list(range(len(BODIES)))
    event = store.record_recall_event(
        query=QUERY,
        scope=SCOPE,
        ambient_context=dict(AMBIENT, transport_session_id="live-transport"),
        results=[
            {
                "node_id": node_ids[index],
                "rank": rank,
                "bm25_score": round(1.0 - 0.1 * rank, 3),
                "vector_score": round(0.8 - 0.05 * rank, 3),
                "graph_score": 0.0,
            }
            for rank, index in enumerate(delivery)
        ],
    )
    return mcp, store, node_ids, event.id


def _usefulness(store: MemoryStore, node_ids: list[str]) -> list[float]:
    return [store.get_node(node_id).usefulness_score for node_id in node_ids]


def _weights(store: MemoryStore, scope: str = SCOPE) -> tuple[float, float, float]:
    weights = store.get_retrieval_weights(scope)
    return (weights.bm25, weights.vector, weights.graph)


def _edge_targets(store: MemoryStore, anchor_id: str) -> dict[str, float]:
    return {
        edge.target_id: edge.weight
        for edge in store.list_query_anchor_edges(anchor_id=anchor_id)
    }


def _containment(verdict: dict[str, Any], node_id: str) -> float:
    return next(entry["containment"] for entry in verdict["results"] if entry["node_id"] == node_id)


# ---------------------------------------------------------------------------
# (a) Overlapping evidence credits what was used, and only that
# ---------------------------------------------------------------------------


def test_overlapping_evidence_credits_only_the_used_node(tmp_path: Path) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "headline.sqlite3")
    before = _usefulness(store, node_ids)
    before_weights = _weights(store)

    verdict = mcp.tools["memory_attest"](
        event_id, [EVICTION_DIFF], context=dict(EXTRACTOR_CONTEXT)
    )

    # The verdict names exactly one grounded result, and it is the one whose
    # content the diff actually reuses.
    assert verdict["grounded_node_ids"] == [node_ids[REDIS]]
    assert verdict["credited"] is True
    assert verdict["replay"] is False
    assert verdict["scope"] == SCOPE
    assert verdict["min_containment"] == ATTESTATION_MIN_CONTAINMENT == 0.25
    assert verdict["evidence_items"] == 1
    assert verdict["evidence_chars"] == len(EVICTION_DIFF.rstrip("\n"))
    assert {entry["node_id"] for entry in verdict["results"]} == set(node_ids)
    assert _containment(verdict, node_ids[REDIS]) > ATTESTATION_MIN_CONTAINMENT
    for entry in verdict["results"]:
        assert entry["evidence_index"] == 0
        if entry["node_id"] != node_ids[REDIS]:
            assert entry["grounded"] is False
            assert entry["containment"] < ATTESTATION_MIN_CONTAINMENT

    # Usefulness moved for the used node and for nothing else.
    after = _usefulness(store, node_ids)
    assert after[REDIS] > before[REDIS]
    for index, (old, new) in enumerate(zip(before, after, strict=True)):
        if index != REDIS:
            assert new == old, f"{BODIES[index]!r} was never used and must not be credited"

    # The learned weights of the *event's* scope moved, by the used result's
    # own per-channel evidence shares — and no other scope's did.
    assert _weights(store) != before_weights
    with MemoryStore(tmp_path / "pristine.sqlite3") as pristine:
        assert _weights(store, "global") == _weights(pristine, "global")

    # One anchor for the event's question, pointing at the grounded subset only.
    anchors = store.list_query_anchors(scope=None, include_decayed=True)
    assert len(anchors) == 1
    assert anchors[0].scope == SCOPE
    assert anchors[0].query == QUERY
    assert verdict["anchor_ids"] == [anchors[0].id]
    assert _edge_targets(store, anchors[0].id) == {
        node_ids[REDIS]: pytest.approx(ANCHOR_EDGE_WEIGHT)
    }
    assert store.count_query_anchor_edges() == 1


def test_credit_signal_decays_with_rank(tmp_path: Path) -> None:
    """Two results used equally, delivered at ranks 0 and 3, earn 1.0 vs 0.25.

    Both nodes start from identical usefulness and access counts, so the
    diminishing-returns gain is the same for both and the whole difference in
    the increment is the ``max(0.2, 1 / (rank + 1))`` decay the live path
    applies. Delivering the same evidence must not make a rank-4 result worth
    as much as a rank-1 one.
    """

    order = [REDIS, 1, 4, ALEMBIC, 2, 5, 6, 7]
    mcp, store, node_ids, event_id = _seed(tmp_path / "rank.sqlite3", order=order)
    before = _usefulness(store, node_ids)
    assert before[REDIS] == before[ALEMBIC] == 0.0
    assert store.get_node(node_ids[REDIS]).access_count == (
        store.get_node(node_ids[ALEMBIC]).access_count
    )

    verdict = mcp.tools["memory_attest"](
        event_id, [EVICTION_DIFF, ALEMBIC_OUTPUT], context=dict(EXTRACTOR_CONTEXT)
    )

    assert set(verdict["grounded_node_ids"]) == {node_ids[REDIS], node_ids[ALEMBIC]}
    # Each node's containment is credited to the item that produced it.
    indices = {entry["node_id"]: entry["evidence_index"] for entry in verdict["results"]}
    assert indices[node_ids[REDIS]] == 0
    assert indices[node_ids[ALEMBIC]] == 1
    ranks = {entry["node_id"]: entry["rank"] for entry in verdict["results"]}
    assert ranks[node_ids[REDIS]] == 0
    assert ranks[node_ids[ALEMBIC]] == 3

    after = _usefulness(store, node_ids)
    top = after[REDIS] - before[REDIS]
    fourth = after[ALEMBIC] - before[ALEMBIC]
    assert top > fourth > 0.0
    # signal(rank 0) = 1.0, signal(rank 3) = max(0.2, 1/4) = 0.25.
    assert top / fourth == pytest.approx(4.0)

    anchor = store.list_query_anchors(scope=SCOPE, include_decayed=True)[0]
    assert set(_edge_targets(store, anchor.id)) == {node_ids[REDIS], node_ids[ALEMBIC]}


# ---------------------------------------------------------------------------
# (b) Negative control: real artifacts from another task earn nothing
# ---------------------------------------------------------------------------


def test_realistic_unrelated_evidence_earns_nothing(tmp_path: Path) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "negative.sqlite3")
    before = _usefulness(store, node_ids)
    before_weights = _weights(store)

    # The control is only worth anything if the evidence is the size and shape
    # a session really produces. Expressed against the cap rather than against
    # a literal, so re-fitting the caps cannot quietly shrink the control into
    # a toy: every item is most of a full graded unit, and the submission is
    # three of them.
    assert all(
        len(item) > 0.7 * EVIDENCE_MAX_ITEM_CHARS for item in UNRELATED_EVIDENCE
    )
    assert (
        sum(len(item) for item in UNRELATED_EVIDENCE) > 2.5 * EVIDENCE_MAX_ITEM_CHARS
    )

    verdict = mcp.tools["memory_attest"](
        event_id, list(UNRELATED_EVIDENCE), context=dict(EXTRACTOR_CONTEXT)
    )

    assert verdict["grounded_node_ids"] == []
    assert verdict["credited"] is False
    assert verdict["anchor_ids"] == []
    assert verdict["evidence_items"] == 3
    assert all(entry["grounded"] is False for entry in verdict["results"])
    # Not merely under the gate — nowhere near it.
    assert max(entry["containment"] for entry in verdict["results"]) < 0.5 * ATTESTATION_MIN_CONTAINMENT

    assert _usefulness(store, node_ids) == before
    assert _weights(store) == before_weights
    assert store.list_query_anchors(scope=None, include_decayed=True) == []
    assert store.count_query_anchor_edges() == 0

    event = store.get_recall_event(event_id)
    assert event.feedback_applied is False
    assert event.feedback_trace_id is None

    # A graded-and-rejected submission is still recorded, so a re-run replays
    # rather than re-grading.
    ledger = store.list_recall_attestations(recall_event_id=event_id)
    assert len(ledger) == 1
    assert ledger[0].credited is False
    assert ledger[0].source_session_key == EXTRACTOR_CONTEXT["source_session_key"]
    assert ledger[0].agent == "extractor"


def test_evidence_is_graded_per_item_never_concatenated(tmp_path: Path) -> None:
    """Fragments that only ground when joined must not ground.

    The five items come from one real cert-manager session and each carries a
    couple of node 2's tokens. Joined they clear the gate by a wide margin;
    individually none reaches half of it. Grading a single concatenated
    document would credit a node that no artifact in the session evidences —
    the accidental grounding the 0.25 threshold cannot survive at document
    scale.
    """

    mcp, store, node_ids, event_id = _seed(tmp_path / "fragments.sqlite3")
    before = _usefulness(store, node_ids)

    verdict = mcp.tools["memory_attest"](
        event_id, list(FRAGMENTED_EVIDENCE), context=dict(EXTRACTOR_CONTEXT)
    )

    assert verdict["grounded_node_ids"] == []
    assert verdict["credited"] is False
    assert _usefulness(store, node_ids) == before

    # The premise: joined, this very evidence would have been credited.
    from living_memory.grounding import ground_results

    joined = ground_results(
        "\n".join(FRAGMENTED_EVIDENCE),
        {node_ids[CERT_MANAGER]: store.get_node(node_ids[CERT_MANAGER]).content},
        min_containment=ATTESTATION_MIN_CONTAINMENT,
    )
    assert joined[node_ids[CERT_MANAGER]].grounded is True
    assert _containment(verdict, node_ids[CERT_MANAGER]) < ATTESTATION_MIN_CONTAINMENT


# ---------------------------------------------------------------------------
# (c) A client-asserted verdict is not read
# ---------------------------------------------------------------------------


def test_client_asserted_verdict_changes_nothing(tmp_path: Path) -> None:
    """The same submission, one copy screaming "everything is grounded".

    Two byte-identical databases (SQLite's own backup, so node, event and
    anchor ids match exactly) are attested with the same event and the same
    evidence. One carries a context stuffed with the verdict fields a client
    might hope the server honours — per-result ``grounded``/``containment``, a
    lowered ``min_containment``, a full ``grounded_node_ids`` list. The outputs
    must be identical apart from the fresh attestation id, and so must the
    resulting database state.

    The anchor is created before the copy, by an ordinary live consumption of
    an earlier recall of the same question, so that even ``anchor_ids`` is
    comparable rather than a fresh identifier on each side.
    """

    honest_db = tmp_path / "honest.sqlite3"
    mcp = create_mcp_server(honest_db, mcp_factory=FakeMCP, expose_attest=True)
    store: MemoryStore = mcp.memory_store
    node_ids = [
        mcp.tools["memory_remember"](
            f"{QUERY} {body}", {"scope": SCOPE, **AMBIENT}
        )["node"]["id"]
        for body in BODIES
    ]
    # A live grounded consumption of an earlier recall: creates the anchor for
    # this question, so both runs below reinforce the same anchor row.
    mcp.tools["memory_recall"](
        QUERY, scope=SCOPE, max_results=len(BODIES), depth=0, ambient_context=dict(AMBIENT)
    )
    mcp.tools["memory_remember"](
        f"{QUERY} follow-up: the {BODIES[REDIS]} note held up", {"scope": SCOPE, **AMBIENT}
    )
    anchor_id = store.list_query_anchors(scope=SCOPE, include_decayed=True)[0].id
    event_id = store.record_recall_event(
        query=QUERY,
        scope=SCOPE,
        ambient_context=dict(AMBIENT),
        results=[
            {"node_id": node_id, "bm25_score": 0.9, "vector_score": 0.7, "graph_score": 0.0}
            for node_id in node_ids
        ],
    ).id

    spoofed_db = tmp_path / "spoofed.sqlite3"
    destination = sqlite3.connect(str(spoofed_db))
    try:
        store.connection.backup(destination)
    finally:
        destination.close()
    spoofed = create_mcp_server(spoofed_db, mcp_factory=FakeMCP, expose_attest=True)

    poisoned = dict(
        EXTRACTOR_CONTEXT,
        grounded=True,
        useful=True,
        containment=1.0,
        min_containment=0.0,
        credited=True,
        grounded_node_ids=list(node_ids),
        results=[
            {"node_id": node_id, "grounded": True, "containment": 1.0, "rank": 0}
            for node_id in node_ids
        ],
    )
    honest_verdict = mcp.tools["memory_attest"](
        event_id, [EVICTION_DIFF], context=dict(EXTRACTOR_CONTEXT)
    )
    spoofed_verdict = spoofed.tools["memory_attest"](
        event_id, [EVICTION_DIFF], context=poisoned
    )

    # The one field that cannot match is the attestation's own fresh id.
    assert honest_verdict.pop("attestation_id") != spoofed_verdict.pop("attestation_id")
    assert json.dumps(spoofed_verdict, sort_keys=True) == json.dumps(
        honest_verdict, sort_keys=True
    )
    assert honest_verdict["grounded_node_ids"] == [node_ids[REDIS]]
    assert honest_verdict["anchor_ids"] == [anchor_id]

    # And the state each one left behind is identical too.
    spoofed_store: MemoryStore = spoofed.memory_store
    assert _usefulness(spoofed_store, node_ids) == _usefulness(store, node_ids)
    assert _weights(spoofed_store) == _weights(store)
    assert _edge_targets(spoofed_store, anchor_id) == _edge_targets(store, anchor_id)


def test_credit_policy_env_does_not_reach_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``all`` must not credit the delivered set; ``grounded_negative`` must not blame.

    Attestation grades bounded, redacted evidence. ``all`` would restore the
    vacuous "everything delivered was useful" signal, and absence from a capped
    evidence sample is not evidence that a node went unused.
    """

    for policy in ("all", "grounded_negative"):
        monkeypatch.setenv("LM_RECALL_CREDIT_POLICY", policy)
        mcp, store, node_ids, event_id = _seed(tmp_path / f"policy-{policy}.sqlite3")
        before = _usefulness(store, node_ids)

        verdict = mcp.tools["memory_attest"](
            event_id, [EVICTION_DIFF], context=dict(EXTRACTOR_CONTEXT)
        )

        assert verdict["grounded_node_ids"] == [node_ids[REDIS]], policy
        after = _usefulness(store, node_ids)
        assert after[REDIS] > before[REDIS], policy
        for index, (old, new) in enumerate(zip(before, after, strict=True)):
            if index != REDIS:
                assert new == old, f"{policy} moved an ungrounded result"
        assert store.count_query_anchor_edges() == 1, policy


# ---------------------------------------------------------------------------
# (d) The ledger is the replay guard
# ---------------------------------------------------------------------------


def test_reattesting_the_same_evidence_applies_nothing(tmp_path: Path) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "replay.sqlite3")

    first = mcp.tools["memory_attest"](
        event_id, [EVICTION_DIFF], context=dict(EXTRACTOR_CONTEXT)
    )
    anchor = store.list_query_anchors(scope=SCOPE, include_decayed=True)[0]
    after_first = _usefulness(store, node_ids)
    weights_first = _weights(store)
    edges_first = {
        edge.target_id: (edge.weight, edge.hits)
        for edge in store.list_query_anchor_edges(anchor_id=anchor.id)
    }

    # Same evidence, differently framed on the wire: canonicalization strips
    # trailing whitespace and blank edges, so this is the same digest.
    second = mcp.tools["memory_attest"](
        event_id,
        ["\n" + EVICTION_DIFF.replace("\n", "   \n") + "\n\n"],
        context=dict(EXTRACTOR_CONTEXT, session_id="extraction-run-2"),
    )

    assert second["replay"] is True
    assert second["evidence_sha256"] == first["evidence_sha256"]
    assert second == dict(first, replay=True)

    assert _usefulness(store, node_ids) == after_first
    assert _weights(store) == weights_first
    assert {
        edge.target_id: (edge.weight, edge.hits)
        for edge in store.list_query_anchor_edges(anchor_id=anchor.id)
    } == edges_first
    assert len(store.list_recall_attestations(recall_event_id=event_id)) == 1

    # Different evidence for the same event is a different submission.
    third = mcp.tools["memory_attest"](
        event_id, [ALEMBIC_OUTPUT], context=dict(EXTRACTOR_CONTEXT)
    )
    assert third["replay"] is False
    assert third["evidence_sha256"] != first["evidence_sha256"]
    assert third["grounded_node_ids"] == [node_ids[ALEMBIC]]
    assert len(store.list_recall_attestations(recall_event_id=event_id)) == 2


# ---------------------------------------------------------------------------
# (e) + (f) Closure is a decision of its own
# ---------------------------------------------------------------------------


def test_trace_id_closes_the_event_and_credits_linkage_once(tmp_path: Path) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "closure.sqlite3")
    fingerprint = store.get_recall_event(event_id)
    extracted = store.append_trace(
        "extractor trace: the eviction fix landed in services/cache/eviction.py",
        {"scope": SCOPE, "agent": "extractor"},
    )
    before = store.get_recall_fingerprint_stats(
        store.connection.execute(
            "SELECT fingerprint FROM recall_events WHERE id = ?", (event_id,)
        ).fetchone()["fingerprint"]
    )
    assert fingerprint.feedback_applied is False
    assert before.linked_count == 0

    verdict = mcp.tools["memory_attest"](
        event_id, [EVICTION_DIFF], context=dict(EXTRACTOR_CONTEXT), trace_id=extracted.id
    )

    assert verdict["closed"] is True
    assert verdict["closed_by_attestation"] is True
    assert verdict["feedback_trace_id"] == extracted.id
    event = store.get_recall_event(event_id)
    assert event.feedback_applied is True
    assert event.feedback_trace_id == extracted.id
    key = store.connection.execute(
        "SELECT fingerprint FROM recall_events WHERE id = ?", (event_id,)
    ).fetchone()["fingerprint"]
    assert store.get_recall_fingerprint_stats(key).linked_count == before.linked_count + 1

    # A second attestation of the same event — different evidence, so not a
    # replay — still credits, but must not re-close: re-marking would take the
    # legacy overwrite branch and clobber the pointer above.
    other = store.append_trace("a second extractor trace", {"scope": SCOPE})
    again = mcp.tools["memory_attest"](
        event_id, [ALEMBIC_OUTPUT], context=dict(EXTRACTOR_CONTEXT), trace_id=other.id
    )

    assert again["replay"] is False
    assert again["credited"] is True, "evidence and a consuming trace are different observations"
    assert again["closed"] is True
    assert again["closed_by_attestation"] is False
    assert again["feedback_trace_id"] == extracted.id
    assert store.get_recall_event(event_id).feedback_trace_id == extracted.id
    assert store.get_recall_fingerprint_stats(key).linked_count == before.linked_count + 1


def test_attestation_without_trace_id_credits_without_closing(tmp_path: Path) -> None:
    """The default. Closing needs a real node; inventing one would be a lie.

    ``recall_events.feedback_trace_id`` references ``nodes(id)``, so a
    synthetic node would have to be written to close an event that produced no
    trace — polluting ``nodes`` with non-knowledge and inflating
    ``recall_fingerprints.linked_count``, the very measure this path exists to
    make honest.
    """

    mcp, store, node_ids, event_id = _seed(tmp_path / "open.sqlite3")
    key = store.connection.execute(
        "SELECT fingerprint FROM recall_events WHERE id = ?", (event_id,)
    ).fetchone()["fingerprint"]

    verdict = mcp.tools["memory_attest"](
        event_id, [EVICTION_DIFF], context=dict(EXTRACTOR_CONTEXT)
    )

    assert verdict["credited"] is True
    assert verdict["closed"] is False
    assert verdict["closed_by_attestation"] is False
    assert verdict["feedback_trace_id"] is None
    event = store.get_recall_event(event_id)
    assert event.feedback_applied is False
    assert event.feedback_trace_id is None
    assert store.get_recall_fingerprint_stats(key).linked_count == 0
    # Credit really was applied; only closure was withheld.
    assert store.get_node(node_ids[REDIS]).usefulness_score > 0.0


def test_unresolvable_trace_id_is_rejected_not_downgraded(tmp_path: Path) -> None:
    mcp, store, node_ids, event_id = _seed(tmp_path / "bad-trace.sqlite3")
    before = _usefulness(store, node_ids)

    with pytest.raises(AttestationError, match="resolves to no node"):
        mcp.tools["memory_attest"](
            event_id, [EVICTION_DIFF], context=dict(EXTRACTOR_CONTEXT), trace_id="01NOSUCHNODE"
        )

    assert _usefulness(store, node_ids) == before
    assert store.list_recall_attestations(recall_event_id=event_id) == []
    assert store.count_query_anchor_edges() == 0


def test_unknown_recall_event_is_rejected(tmp_path: Path) -> None:
    mcp, _store, _node_ids, _event_id = _seed(tmp_path / "unknown.sqlite3")

    with pytest.raises(AttestationError, match="unknown recall_event_id"):
        mcp.tools["memory_attest"]("01NOSUCHEVENT", [EVICTION_DIFF])


def test_results_whose_nodes_are_gone_are_skipped(tmp_path: Path) -> None:
    """A decayed or forgotten result is ordinary, not a reason to fail."""

    mcp, store, node_ids, event_id = _seed(tmp_path / "gone.sqlite3")
    store.connection.execute("DELETE FROM nodes WHERE id = ?", (node_ids[1],))
    store.connection.commit()

    verdict = mcp.tools["memory_attest"](event_id, [EVICTION_DIFF])

    assert {entry["node_id"] for entry in verdict["results"]} == set(node_ids) - {node_ids[1]}
    assert verdict["grounded_node_ids"] == [node_ids[REDIS]]


# ---------------------------------------------------------------------------
# Canonicalization: a contract the offline client imports
# ---------------------------------------------------------------------------


def test_canonical_evidence_normalizes_and_digests() -> None:
    items, digest = canonical_evidence(
        ["\n\n  line one   \r\n\tline two\t\t\n\n", "", "   \n  \n", "second item\r\n"]
    )

    assert items == ("  line one\n\tline two", "second item")
    assert canonical_evidence(["  line one\n\tline two", "second item"]) == (items, digest)
    # The digest is over the canonical items joined by RECORD SEPARATOR, so a
    # different split of the same text is a different submission.
    assert canonical_evidence(["  line one\n\tline two\nsecond item"])[1] != digest
    assert len(digest) == 64
    assert ATTESTATION_MIN_CONTAINMENT == RECALL_CREDIT_MIN_CONTAINMENT == 0.25


@pytest.mark.parametrize(
    ("evidence", "cap"),
    [
        (["x"] * (EVIDENCE_MAX_ITEMS + 1), "EVIDENCE_MAX_ITEMS"),
        (["x" * (EVIDENCE_MAX_ITEM_CHARS + 1)], "EVIDENCE_MAX_ITEM_CHARS"),
        (
            ["y" * EVIDENCE_MAX_ITEM_CHARS] * (EVIDENCE_MAX_TOTAL_CHARS // EVIDENCE_MAX_ITEM_CHARS + 1),
            "EVIDENCE_MAX_TOTAL_CHARS",
        ),
    ],
)
def test_over_cap_evidence_is_rejected_by_name(evidence: list[str], cap: str) -> None:
    """Reject, never truncate: a client must not believe more was graded."""

    with pytest.raises(AttestationError, match=cap):
        canonical_evidence(evidence)


def test_evidence_with_nothing_gradable_is_rejected(tmp_path: Path) -> None:
    mcp, store, _node_ids, event_id = _seed(tmp_path / "empty.sqlite3")

    with pytest.raises(AttestationError, match="no gradable item"):
        mcp.tools["memory_attest"](event_id, ["", "  \n\t\n  "])
    with pytest.raises(AttestationError, match="not a single string"):
        mcp.tools["memory_attest"](event_id, EVICTION_DIFF)
    assert store.list_recall_attestations(recall_event_id=event_id) == []


# ---------------------------------------------------------------------------
# (g) Schema v8: the ledger appears on reopen
# ---------------------------------------------------------------------------


def test_v7_era_database_gains_the_ledger_on_reopen(tmp_path: Path) -> None:
    """``_initialize_schema`` runs CREATE TABLE IF NOT EXISTS on every open.

    That is why there is no ``_migrate_pre_v8_schema``: a file written before
    the ledger existed gains it by being reopened, with every existing row
    intact. Dropping the table from a live file reproduces exactly that state.
    """

    db = tmp_path / "legacy.sqlite3"
    mcp, store, node_ids, event_id = _seed(db)
    mcp.tools["memory_attest"](event_id, [EVICTION_DIFF], context=dict(EXTRACTOR_CONTEXT))
    usefulness = _usefulness(store, node_ids)
    store.connection.execute(f"DROP TABLE {RECALL_ATTESTATION_TABLE}")
    store.connection.execute("UPDATE metadata SET value = '7' WHERE key = 'schema_version'")
    store.connection.commit()
    store.close()

    with MemoryStore(db) as reopened:
        tables = {
            str(row["name"])
            for row in reopened.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert RECALL_ATTESTATION_TABLE in tables
        assert reopened.connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()["value"] == "8"

        # No data loss: nodes, the recall event, the anchor and its edges all
        # survive, and the ledger is simply empty again.
        assert _usefulness(reopened, node_ids) == usefulness
        assert reopened.get_recall_event(event_id) is not None
        assert reopened.count_query_anchor_edges() == 1
        assert reopened.list_recall_attestations() == []

        # And the first thing it can do is grade a submission it has forgotten.
        replayed = attest_recall_usage(
            reopened, event_id, [EVICTION_DIFF], context=dict(EXTRACTOR_CONTEXT)
        )
        assert replayed["replay"] is False
        assert replayed["grounded_node_ids"] == [node_ids[REDIS]]

    # Idempotent: reopening again neither duplicates nor drops the table.
    with MemoryStore(db) as again:
        assert len(again.list_recall_attestations()) == 1
        assert (
            int(
                again.connection.execute(
                    "SELECT COUNT(*) AS count FROM sqlite_master WHERE name = ?",
                    (RECALL_ATTESTATION_TABLE,),
                ).fetchone()["count"]
            )
            == 1
        )
