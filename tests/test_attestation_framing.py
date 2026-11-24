"""Evidence boundaries, durable replay and extraction agree on synthetic inputs."""

import asyncio
import hashlib
import json
import sqlite3
from types import SimpleNamespace

from fastmcp import Client
from fastmcp.exceptions import ToolError
import pytest

from living_memory import attestation
from living_memory.postsession.evidence import assemble_evidence, proposed_attestations
from living_memory.postsession.judge import canonical_json
from living_memory.postsession.runner import (
    ExtractionRunner, Ledger, RunnerConfig, op_fingerprint,
)
from living_memory.postsession.session import (
    Evidence, FileMutation, ProposedOp, RecallInteraction, SessionRecord, SourceSpan, ToolCall,
)
from living_memory.server import create_mcp_server


WORDS = ["cobalt", "zircon", "amber", "fuchsia", "orchid", "quartz", "saffron", "indigo"]
JOINED = "\x1e".join(WORDS)
DIGEST = "fd691630df8c1607beb8c8441ada4b2666d03fce32144d02d8f424e808675edf"
SCOPE = "project:synthetic-framing"


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    for key, value in {
        "LIVING_MEMORY_EMBEDDING_BACKEND": "hash",
        "LIVING_MEMORY_CHUNK_TOKENIZER": "heuristic",
        "LM_AUTO_CONSOLIDATE_POLICY": "off",
        "LM_RETRIEVAL_TUNING_POLICY": "fixed",
        "LM_LOOKUP_CREDIT_POLICY": "off",
    }.items():
        monkeypatch.setenv(key, value)
    mcp = create_mcp_server(tmp_path / "memory.sqlite3", expose_attest=True)
    store = mcp.memory_store
    node = store.create_node(level="trace", content=" ".join(WORDS),
                             context={"scope": SCOPE}, embedding=[1.0, 0.0])
    event = store.record_recall_event(query="", scope=SCOPE, results=[{
        "node_id": node.id, "bm25_score": 1.0,
    }])
    try:
        yield SimpleNamespace(mcp=mcp, store=store, node=node, event=event)
    finally:
        mcp.auto_consolidation.close()
        store.close()


def snapshot(store):
    with sqlite3.connect(store.db_path) as reader:
        return tuple(reader.iterdump())


def attest(r, evidence, route="direct"):
    async def native():
        async with Client(r.mcp) as client:
            return (await client.call_tool("memory_attest", {
                "recall_event_id": r.event.id, "evidence": evidence,
            })).data

    if route == "native":
        return asyncio.run(native())
    return attestation.attest_recall_usage(r.store, r.event.id, evidence)


def old_identity(items):
    # These fixture inputs are already canonical and below every cap.
    return tuple(items), hashlib.sha256("\x1e".join(items).encode()).hexdigest()


@pytest.mark.parametrize(("raw", "canonical", "digest"), [
    ([], (), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    (["alpha", "beta"], ("alpha", "beta"),
     "1841875bc4281532fcbfb57337ec4ed6f0c4d9348e8755825f31acf5fd04a8e0"),
    (["\nAlpha  \r\n  beta\t\r\n", "", "Gamma\rdelta\r\n"],
     ("Alpha\n  beta", "Gamma\ndelta"),
     "46010174319f753c545310fd6242b6a9e2eeb83125f0d2befc2aec05a4a4de89"),
    (["x" * 600], ("x" * 600,),
     "5130b33e6b87fbf5316ed9049e98924eb110800bcbaaad8050f642fba6df37c9"),
    (["y" * 600] * 32, ("y" * 600,) * 32,
     "4b04749cf88dbdacc212506e810496ff0622f2ed46d38a344df760e37960528c"),
    (["\x1e", "alpha\x1e\n", "\n"], ("alpha",),
     "8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8"),
])
def test_published_canonical_digests(raw, canonical, digest):
    assert attestation.canonical_evidence(raw) == (canonical, digest)


@pytest.mark.parametrize("item", [JOINED, "\x1ealpha", "alpha\n\x1ebeta", "alpha\x1e beta"])
def test_canonical_separator_is_explicitly_rejected(item):
    with pytest.raises(attestation.AttestationError, match="item 1.*U\\+001E"):
        attestation.canonical_evidence(["", item])


@pytest.mark.parametrize("route", ["direct", "native"])
@pytest.mark.parametrize("prior", [False, True])
def test_separator_rejected_before_preparation_or_ledger(runtime, monkeypatch, route, prior):
    r = runtime
    if prior:
        result = attest(r, WORDS)
        assert not result["credited"]
        assert result["results"][0]["containment"] < attestation.ATTESTATION_MIN_CONTAINMENT
    before = snapshot(r.store)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid evidence reached preparation, grading or the ledger")

    monkeypatch.setattr(attestation.feedback, "prepared_feedback", forbidden)
    monkeypatch.setattr(attestation, "_grade", forbidden)
    monkeypatch.setattr(r.store, "find_recall_attestation", forbidden)
    with pytest.raises((attestation.AttestationError, ToolError), match="U\\+001E"):
        attest(r, [JOINED], route)
    assert snapshot(r.store) == before


@pytest.mark.parametrize("route", ["direct", "native"])
@pytest.mark.parametrize("completed", [False, True])
def test_ambiguous_saved_receipt_cannot_replay_for_ordinary_items(
    runtime, monkeypatch, route, completed,
):
    r = runtime
    if completed:
        with monkeypatch.context() as patch:
            patch.setattr(attestation, "canonical_evidence", old_identity)
            result = attest(r, [JOINED])
        assert result["credited"] and result["results"][0]["containment"] == 1.0
    else:
        r.store.claim_recall_attestation(
            recall_event_id=r.event.id, evidence_sha256=DIGEST,
            evidence_items=1, evidence_chars=len(JOINED),
            min_containment=attestation.ATTESTATION_MIN_CONTAINMENT,
        )
    before = snapshot(r.store)
    with pytest.raises((attestation.AttestationError, ToolError), match="boundaries"):
        attest(r, WORDS, route)
    assert snapshot(r.store) == before


@pytest.mark.parametrize("route", ["direct", "native"])
@pytest.mark.parametrize("evidence", [WORDS, [" ".join(WORDS)]])
def test_identical_canonical_replay_retains_old_digest_and_effects(runtime, monkeypatch, route, evidence):
    r = runtime
    with monkeypatch.context() as patch:
        patch.setattr(attestation, "canonical_evidence", old_identity)
        first = attest(r, evidence)
    before = snapshot(r.store)
    repeated = attest(r, [f"\r\n{item} \t\r\n" for item in evidence], route)
    assert repeated == {**first, "replay": True}
    assert repeated["evidence_sha256"] == old_identity(evidence)[1]
    assert snapshot(r.store) == before


def record(event_id="synthetic-event"):
    session = SessionRecord(session_key="claude:framing", source="claude", cli="claude",
                            path="/synthetic/session.jsonl", transcript_sha256="a" * 64)
    span = SourceSpan("claude", session.path, 1)
    session.recalls.append(RecallInteraction(ordinal=0, query="", span=session.span(0),
                                             recall_event_id=event_id))
    return session, span


@pytest.mark.parametrize(("items", "cap"), [
    ([JOINED] + [""] * 64, "EVIDENCE_MAX_ITEMS"),
    ([JOINED, "x" * 601], "EVIDENCE_MAX_ITEM_CHARS"),
    ([JOINED] + ["y" * 600] * 32, "EVIDENCE_MAX_TOTAL_CHARS"),
])
def test_over_cap_fingerprint_remains_stable_with_reserved_separator(items, cap):
    session, span = record()
    op = ProposedOp.for_session("attest", {
        "recall_event_id": "synthetic-event", "evidence": items,
    }, session, span, [Evidence("synthetic command", span.locator())])
    with pytest.raises(attestation.AttestationError, match=cap):
        attestation.canonical_evidence(items)
    fallback = hashlib.sha256("\x1f".join(item.strip() for item in items if item.strip()).encode()).hexdigest()
    expected = hashlib.sha256(canonical_json({
        "kind": "attest", "recall_event_id": "synthetic-event", "evidence_sha256": fallback,
    }, indent=None).encode()).hexdigest()
    assert op_fingerprint(op) == expected


@pytest.mark.parametrize("source", ["command", "diff", "file"])
def test_extraction_reports_invalid_evidence_without_proposing_or_writing(tmp_path, source):
    session, span = record()
    if source == "command":
        session.tool_calls.append(ToolCall(0, "exec_command", None, {"cmd": "probe"}, span,
                                           result_text=JOINED, result_span=session.span(2)))
    else:
        fields = {"post_image": JOINED} if source == "file" else {
            "unified_diff": f"--- a/sample\n+++ b/sample\n@@ -0,0 +1 @@\n+{JOINED}\n",
        }
        session.file_mutations.append(FileMutation(0, "sample", "create", span, action_span=span, **fields))
    with pytest.raises(attestation.AttestationError, match="U\\+001E"):
        assemble_evidence(session)
    ledger = Ledger(tmp_path)
    runner = ExtractionRunner(config=RunnerConfig(stages=("attest",)), ledger=ledger)
    proposals = runner.propose(session)
    assert proposals.attests == []
    assert "U+001E" in proposals.stage_stats["attest"]["error"]
    assert len(proposals.errors) == 1
    assert runner.decide_and_execute([proposals]) == []
    assert ledger.entries() == []


def test_extraction_client_and_server_use_the_same_canonical_digest(runtime):
    session, span = record(runtime.event.id)
    session.tool_calls.append(ToolCall(0, "exec_command", None, {"cmd": "probe"}, span,
                                       result_span=session.span(2), result_text="\r\n" + " ".join(WORDS) + " \t\r\n"))
    bundle = assemble_evidence(session)
    op = proposed_attestations(session, bundle=bundle)[0]
    result = attest(runtime, op.payload["evidence"], "native")
    assert result["evidence_sha256"] == bundle.evidence_sha256
    assert tuple(op.payload["evidence"]) == bundle.texts
    assert result["credited"]
    before = snapshot(runtime.store)
    assert attest(runtime, op.payload["evidence"], "native")["replay"]
    assert snapshot(runtime.store) == before


@pytest.mark.parametrize("evidence", [WORDS, [JOINED]])
def test_saved_unknown_identity_survives_rejection(runtime, tmp_path, evidence):
    session, span = record(runtime.event.id)
    op = ProposedOp.for_session("attest", {
        "recall_event_id": runtime.event.id, "evidence": evidence,
    }, session, span, [Evidence("synthetic command", span.locator())]).validate()
    fingerprint = hashlib.sha256(canonical_json({
        "kind": "attest", "recall_event_id": runtime.event.id, "evidence_sha256": DIGEST,
    }, indent=None).encode()).hexdigest()
    state = tmp_path / "ledger"
    state.mkdir()
    saved = {
        "transcript_sha256": session.transcript_sha256, "fingerprint": fingerprint,
        "kind": "attest", "session_key": session.session_key, "operation": op.to_dict(),
        "recipient": "synthetic://recipient", "state": "unknown",
    }
    path = state / "ledger.jsonl"
    path.write_text(json.dumps(saved) + "\n")
    runtime.store.claim_recall_attestation(
        recall_event_id=runtime.event.id, evidence_sha256=DIGEST,
        evidence_items=1, evidence_chars=len(JOINED),
        min_containment=attestation.ATTESTATION_MIN_CONTAINMENT,
    )
    before = snapshot(runtime.store)
    assert op_fingerprint(op) == fingerprint
    ledger = Ledger(state)

    class Recipient:
        url = "synthetic://recipient"

        def attest(self, event_id, items, **kwargs):
            return attestation.attest_recall_usage(runtime.store, event_id, items, **kwargs)

    runner = ExtractionRunner(config=RunnerConfig(dry_run=False), ledger=ledger, client=Recipient())
    decision = runner.reconcile()[0]
    assert (decision.status, decision.reason) == ("unknown", "unconfirmed_effect")
    assert decision.fingerprint == fingerprint
    assert Ledger(state).entries() == [saved]
    assert snapshot(runtime.store) == before
