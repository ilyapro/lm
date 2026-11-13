"""Source-aware execution and saved-operation compatibility, without a corpus or judge."""

import copy
import hashlib
import json
import sqlite3
from dataclasses import replace

import pytest

from living_memory.postsession.client import ClientError, LiveMemoryClient, PreDispatchError
from living_memory.postsession.runner import (
    ExtractionRunner, Ledger, RunnerConfig, SessionProposals, enforce_attribution,
    op_fingerprint,
)
from living_memory.postsession.session import Evidence, SourceSpan
from postsession_support import TOKEN, live_server, make_record, make_remember
from test_postsession_unknown import Recipient


FACT = "config/parser.toml preserves the source of each accepted observation."


def proposal(record, line=3, **context):
    op = make_remember(record, FACT)
    span = SourceSpan(record.source, record.path, line).locator()
    return replace(
        op, provenance=replace(op.provenance, source_span=span),
        payload={**op.payload, "context": {**op.payload["context"], **context}},
        evidence=(Evidence(FACT, span),),
    ).validate()


def legacy_fingerprint(op):
    identity = {
        "kind": "remember", "content": op.payload["content"],
        "scope": op.payload["context"]["scope"],
    }
    return hashlib.sha256(json.dumps(
        identity, sort_keys=True, ensure_ascii=False,
    ).encode()).hexdigest()


def saved_operation(record):
    op = enforce_attribution(proposal(record, note="saved attribution"), record, task="saved task")
    op.payload["context"]["extraction_fingerprint"] = legacy_fingerprint(op)
    return op


def save_legacy(path, op, client, state, *, result_id="", request=True):
    entry = {
        "transcript_sha256": op.provenance.transcript_sha256,
        "fingerprint": legacy_fingerprint(op), "kind": "remember",
        "session_key": op.provenance.session_key, "result_id": result_id,
    }
    if request:
        entry.update(operation=op.to_dict(), recipient=client.url, state=state)
    else:
        assert state == "applied"  # Original completed rows had no state or request.
    path.mkdir(parents=True, exist_ok=True)
    (path / "ledger.jsonl").write_text(json.dumps(entry) + "\n")
    return entry


def runner(path, client, **config):
    return ExtractionRunner(
        config=RunnerConfig(dry_run=False, **config), ledger=Ledger(path), client=client,
    )


def execute(path, client, record, ops, **config):
    return runner(path, client, **config).decide_and_execute([
        SessionProposals(record=record, ops=ops),
    ])


class ReceiptRecipient(Recipient):
    def __init__(self):
        super().__init__()
        self.lookups = []

    def lookup_nodes(self, ids, **kwargs):
        assert kwargs == {"view": "full"}
        self.lookups.append(ids)
        return {"results": [
            {"id": f"node-{i}", "content": fact, "scope": context["scope"], "context": context}
            for i, (fact, context) in enumerate(self.effects, 1) if f"node-{i}" in ids
        ]}


def test_same_span_retries_and_changed_context_keep_one_effect_per_source(tmp_path):
    record, client = make_record(), ReceiptRecipient()
    first, second = proposal(record, source_span="context noise"), proposal(record, 30)
    decisions = execute(tmp_path, client, record, [first, first, second])
    assert [d.status for d in decisions] == ["written", "skipped", "written"]
    assert decisions[0].node_id == decisions[1].node_id != decisions[2].node_id
    noisy = proposal(record, task="different task", session_id="new session", note="new context")
    restarted = execute(tmp_path, client, record, [noisy, second], task="new runner task")
    assert [d.reason for d in restarted] == ["ledger_duplicate", "ledger_duplicate"]
    assert [context["source_span"] for _, context in client.effects] == [
        first.provenance.source_span, second.provenance.source_span,
    ]
    assert len(Ledger(tmp_path)) == Ledger(tmp_path).written_nodes_total() == 2


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("route", ["proposal", "reconcile"])
def test_ready_retry_uses_exact_saved_request_despite_context_noise(tmp_path, legacy, route):
    record, client = make_record(), ReceiptRecipient()
    old = saved_operation(record)
    if legacy:
        save_legacy(tmp_path, old, client, "ready")
    else:
        client.failure = PreDispatchError("not dispatched")
        assert execute(tmp_path, client, record, [old])[0].reason == "pre_dispatch"
        client.failure = None
    saved = Ledger(tmp_path).entries()[0]
    noisy = proposal(record, task="new task", note="new attribution", source_span="context noise")
    result = (execute(tmp_path, client, record, [noisy]) if route == "proposal"
              else runner(tmp_path, client, task="new task").reconcile())[0]
    assert result.status == "written"
    assert result.fingerprint == saved["fingerprint"]
    assert client.effects == [(FACT, saved["operation"]["payload"]["context"])]
    assert Ledger(tmp_path).entries()[0]["operation"] == saved["operation"]
    assert execute(tmp_path, client, record, [noisy])[0].reason == "ledger_duplicate"
    assert len(client.effects) == len(Ledger(tmp_path)) == 1


def test_legacy_ready_recipient_match_completes_the_saved_key(tmp_path):
    record, client = make_record(), ReceiptRecipient()
    old = saved_operation(record)
    client.remember(FACT, old.payload["context"])
    saved = save_legacy(tmp_path, old, client, "ready")
    client.recall = lambda *args, **kwargs: {
        "results": [{"node": client.lookup_nodes(["node-1"], view="full")["results"][0], "delivery": "full"}],
    }
    decision = execute(tmp_path, client, record, [proposal(record)])[0]
    assert (decision.reason, decision.node_id) == ("lm_prior_extraction", "node-1")
    assert decision.fingerprint == saved["fingerprint"]
    assert len(client.effects) == len(Ledger(tmp_path)) == 1
    retained = Ledger(tmp_path).entries()[0]
    assert retained["state"] == "applied"
    assert retained["operation"] == saved["operation"]


@pytest.mark.parametrize("state", ["applied", "ready", "unknown"])
def test_legacy_saved_request_distinguishes_sources_without_reopening_old_state(tmp_path, state):
    record, client = make_record(), ReceiptRecipient()
    old = saved_operation(record)
    if state != "ready":
        client.remember(FACT, old.payload["context"])
    saved = save_legacy(tmp_path, old, client, state, result_id="node-1" if state == "applied" else "")
    result = execute(tmp_path, client, record, [proposal(record, 30)])[0]
    assert result.status == "written"
    assert Ledger(tmp_path).get(record.transcript_sha256, saved["fingerprint"]) == saved
    retried = execute(tmp_path, client, record, [proposal(record, task="new task")])[0]
    assert retried.status == {"applied": "skipped", "ready": "written", "unknown": "unknown"}[state]
    assert retried.fingerprint == saved["fingerprint"]
    assert len(client.effects) == len(Ledger(tmp_path)) == 2
    assert Ledger(tmp_path).get(record.transcript_sha256, saved["fingerprint"])["operation"] == saved["operation"]


@pytest.mark.parametrize("line", [3, 30])
@pytest.mark.parametrize("receipt_fingerprint", ["legacy", "current", "absent"])
def test_original_applied_row_uses_exact_receipt(tmp_path, line, receipt_fingerprint):
    record, client = make_record(), ReceiptRecipient()
    old = saved_operation(record)
    context = dict(old.payload["context"])
    if receipt_fingerprint == "current":
        context["extraction_fingerprint"] = op_fingerprint(old)
    elif receipt_fingerprint == "absent":
        del context["extraction_fingerprint"]
    client.remember(FACT, context)
    saved = save_legacy(tmp_path, old, client, "applied", result_id="node-1", request=False)
    decision = execute(tmp_path, client, record, [proposal(record, line)])[0]
    assert decision.status == ("skipped" if line == 3 else "written")
    assert client.lookups == [["node-1"]]
    assert len(client.effects) == len(Ledger(tmp_path)) == (1 if line == 3 else 2)
    assert Ledger(tmp_path).get(record.transcript_sha256, saved["fingerprint"]) == {**saved, "state": "applied"}
    again = execute(tmp_path, client, record, [proposal(record, line, task="new task")])[0]
    assert again.reason == "ledger_duplicate"
    assert len(client.effects) == (1 if line == 3 else 2)


@pytest.mark.parametrize("fault", [
    "empty", "unavailable", "id", "content", "scope", "transcript", "span", "correction", "fingerprint",
])
def test_unconfirmed_legacy_receipt_cannot_authorize_another_effect(tmp_path, fault):
    record, client = make_record(), ReceiptRecipient()
    old = saved_operation(record)
    client.remember(FACT, old.payload["context"])
    save_legacy(tmp_path, old, client, "applied", result_id="node-1", request=False)
    before = (tmp_path / "ledger.jsonl").read_bytes()
    lookup = client.lookup_nodes

    def faulty(ids, **kwargs):
        if fault == "empty":
            return {"results": []}
        if fault == "unavailable":
            raise ClientError("recipient unavailable")
        result = copy.deepcopy(lookup(ids, **kwargs))
        node = result["results"][0]
        if fault in {"id", "content", "scope"}:
            node[fault] = "unrelated"
        else:
            key = {"transcript": "source_transcript_sha256", "span": "source_span",
                   "correction": "corrected_node_id", "fingerprint": "extraction_fingerprint"}[fault]
            node["context"][key] = "" if fault == "span" else "unrelated"
        return result

    client.lookup_nodes = faulty
    decisions = execute(tmp_path, client, record, [proposal(record), proposal(record, 30)])
    assert [d.reason for d in decisions] == ["legacy_source_unconfirmed"] * 2
    assert all(d.status == "unknown" for d in decisions)
    assert len(client.effects) == 1
    assert (tmp_path / "ledger.jsonl").read_bytes() == before


def test_legacy_unknown_reconciliation_requires_the_exact_source_and_keeps_its_key(tmp_path):
    record, client = make_record(), ReceiptRecipient()
    old = saved_operation(record)
    client.remember(FACT, old.payload["context"])
    other = enforce_attribution(proposal(record, 30), record, task="other")
    client.remember(FACT, other.payload["context"])
    saved = save_legacy(tmp_path, old, client, "unknown")
    key = f"{saved['transcript_sha256']}:{saved['fingerprint']}"
    run = runner(tmp_path, client)
    assert run.reconcile(operation=key, node_id="node-2")[0].status == "unknown"
    assert run.reconcile(operation=key, node_id="missing")[0].status == "unknown"
    confirmed = run.reconcile(operation=key, node_id="node-1")[0]
    assert (confirmed.status, confirmed.node_id, confirmed.fingerprint) == ("reconciled", "node-1", saved["fingerprint"])
    assert len(client.effects) == 2
    assert len(Ledger(tmp_path)) == 1
    assert Ledger(tmp_path).entries()[0]["operation"] == saved["operation"]
    assert execute(tmp_path, client, record, [proposal(record)])[0].reason == "ledger_duplicate"


@pytest.mark.parametrize("changed_fact", ["result_id", "recipient"])
def test_claim_rechecks_legacy_receipt_against_current_row(tmp_path, changed_fact):
    record, client = make_record(), ReceiptRecipient()
    old = saved_operation(record)
    client.remember(FACT, old.payload["context"])
    saved = save_legacy(tmp_path, old, client, "applied", result_id="node-1", request=False)

    def interleaved(*args, **kwargs):
        # Candidate lookup follows the exact receipt read and precedes claim.
        changed = {**saved, changed_fact: "different-result-or-recipient"}
        (tmp_path / "ledger.jsonl").write_text(json.dumps(changed) + "\n")
        return {"results": []}

    client.recall = interleaved
    decision = execute(tmp_path, client, record, [proposal(record, 30)])[0]
    assert decision.status == "unknown"
    assert len(client.effects) == len(Ledger(tmp_path)) == 1


def test_native_recipient_retains_both_sources_and_restart_never_repeats(live_server, tmp_path):
    client = LiveMemoryClient(live_server["url"], token=TOKEN)
    record = make_record(transcript_sha256="a3" * 32)
    ops = [proposal(record), proposal(record, 30)]
    decisions = execute(tmp_path, client, record, ops)
    assert [d.status for d in decisions] == ["written", "written"]
    assert len({d.node_id for d in decisions}) == 2
    calls = list(client.calls)
    for task in ("restart", "different task"):
        assert [d.reason for d in execute(tmp_path, client, record, ops, task=task)] == ["ledger_duplicate"] * 2
    assert client.calls == calls
    with sqlite3.connect(live_server["db"]) as db:
        rows = db.execute(
            "SELECT content, json_extract(context, '$.source_span') FROM nodes "
            "WHERE json_extract(context, '$.source_transcript_sha256') = ?",
            (record.transcript_sha256,),
        ).fetchall()
    assert sorted(rows) == sorted((FACT, op.provenance.source_span) for op in ops)


@pytest.mark.parametrize("state,with_request", [("applied", False), ("ready", True), ("unknown", True)])
def test_native_legacy_effect_is_preserved_through_retry_and_confirmation(live_server, tmp_path, state, with_request):
    client = LiveMemoryClient(live_server["url"], token=TOKEN)
    record = make_record(transcript_sha256={"applied": "b4", "ready": "c5", "unknown": "d6"}[state] * 32)
    old = saved_operation(record)
    node_id = "" if state == "ready" else client.remember(FACT, old.payload["context"])["node"]["id"]
    saved = save_legacy(tmp_path, old, client, state, result_id=node_id if state == "applied" else "", request=with_request)
    decisions = execute(tmp_path, client, record, [proposal(record, 30), proposal(record, task="new task")])
    assert decisions[0].status == "written"
    assert decisions[1].status == {"applied": "skipped", "ready": "written", "unknown": "reconciled"}[state]
    assert decisions[1].fingerprint == saved["fingerprint"]
    if node_id:
        assert decisions[1].node_id == node_id
    assert [d.reason for d in execute(tmp_path, client, record, [proposal(record), proposal(record, 30)])] == ["ledger_duplicate"] * 2
    with sqlite3.connect(live_server["db"]) as db:
        rows = db.execute(
            "SELECT context FROM nodes WHERE json_extract(context, '$.source_transcript_sha256') = ?",
            (record.transcript_sha256,),
        ).fetchall()
    contexts = [json.loads(row[0]) for row in rows]
    assert len(contexts) == len(Ledger(tmp_path)) == 2
    assert old.payload["context"].items() <= next(c for c in contexts if c["source_span"] == old.provenance.source_span).items()
