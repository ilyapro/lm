"""The regression this whole phase exists for: content past the first 128 tokens.

The encoder is configured with ``max_seq_length = 128``. Before chunking, a node
got exactly one vector, so everything past its 128th token was invisible to the
vector channel -- ``cos(vector of the whole node, vector of its first 128
tokens) = 1.000``, measured on the live corpus. This test builds a node whose
answer sits far past that boundary, asks for it in words that appear nowhere in
the corpus, and requires the node in the top 5.

It fails on the pre-change revision. Verified, not assumed: run against
``8b93e8d`` -- the commit before the max-pool vector channel, with the same
schema v6 storage and the same chunk-writing write path -- the node lands **9th
of 9 results** with ``vector_score`` 0.2208, exactly its whole-node cosine; on
this revision it lands **1st of 9** with 0.5393. Both runs are recorded verbatim
in ``artifacts/harness/phase1-gate.md``.

Why the query is Russian against an English corpus: the FTS channel indexes the
*whole* node, tail included, so a query sharing any rare word with the tail would
be answered by bm25 and would pass on the old code too, proving nothing.
Cross-language wording drives ``bm25_score`` to exactly 0 (asserted below), so
the only channel that can find this node is the one under test. The wording also
avoids every entry of the Russian synonym table in ``embeddings._SYNONYMS``,
which would otherwise expand into English tokens.

Why a subprocess: this test needs the real sentence-transformers encoder, and the
hash backend has no 128-token horizon at all, so under
``LIVING_MEMORY_EMBEDDING_BACKEND=hash`` (what ``scripts/test.sh`` exports) the
premise does not exist. Clearing the variable in-process is not enough --
``embeddings._MODEL_CACHE`` would then hand the real model to later tests that
mock ``sentence_transformers``, and ``torch`` would stay in ``sys.modules`` for
tests that assert it was never imported. Both were observed failing that way.
So the whole model-dependent probe runs in a child process which returns
measurements as JSON, and every assertion lives here, in the parent, which
imports nothing from ``living_memory`` at all.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCOPE = "project:chunk-tail-regression"

#: The first 128 model tokens of the node under test: meeting logistics, and
#: nothing an infrastructure question would ever match.
TAIL_HEAD = (
    "Weekly team sync notes, week 34. The Monday sync moves to 10:30 so the "
    "Lisbon half of the team is not dialling in before breakfast, and the "
    "Thursday design review keeps its old slot. Calendar invites were resent "
    "with the new room; the small room on the fourth floor is booked for the "
    "whole quarter, so nobody has to hunt for a free one again. Two people are "
    "on holiday until the end of the month, so the demo rota shifts by one "
    "person and the retro moderator for the next two sessions is whoever "
    "presented last. Snacks and the coffee rota were discussed at length "
    "again: the kitchen on the third floor is restocked on Wednesdays, and the "
    "office manager asked everyone to write their name on their own mug. "
    "Somebody has to bring a laptop dongle to the Thursday session because the "
    "meeting room projector still wants HDMI. Onboarding buddies were assigned "
    "for the two new joiners starting after the summer break, and the "
    "paperwork for their badges is with facilities. The offsite was moved to "
    "the first week of October and the survey about the venue closes on "
    "Friday, so please fill it in before then.\n\n"
)

#: The part the query is about. It starts at character 1160 of 1871 -- around
#: token 246 of 431, twice the encoder's horizon away from the start.
TAIL_ANSWER = (
    "Root cause of the Tuesday registry outage, written down so the next "
    "person does not have to rediscover it. The TLS certificate that serves "
    "the internal container image mirror expired at 04:12 UTC and every pull "
    "started failing handshake. Renewal is automated with certbot, but the "
    "systemd timer that runs it had been left disabled after the host was "
    "rebuilt in November, and the expiry warning went to a Slack channel that "
    "was archived in the meantime, so no human ever saw it. The remedy is "
    "three things: re-enable the renewal timer on the mirror host, route the "
    "expiry warning to the on-call rotation instead of a channel, and add a "
    "synthetic pull that starts failing fourteen days before the certificate "
    "runs out so the alarm arrives while there is still time to act."
)

TAIL_CONTENT = TAIL_HEAD + TAIL_ANSWER

#: Terms that carry the answer. Each one must occur only inside the invisible
#: tail -- the test checks that rather than trusting the prose above.
ANSWER_TERMS = ("certificate", "certbot", "registry", "mirror", "expired")

QUERY = (
    "из-за чего перестал работать внутренний реестр образов "
    "и как починить просроченный сертификат зеркала"
)

#: Plausible neighbours, every one of them about the same infrastructure. They
#: are what fills the top 5 when the tail node is invisible, and several of them
#: out-score a head-only vector of the node under test.
DISTRACTORS = (
    "The build pipeline pushes every merge to main into the internal registry "
    "mirror; warming the layer cache afterwards takes about four minutes.",
    "Image pulls that failed during the Tuesday window were retried "
    "automatically by the deployment job, so no manual action was needed.",
    "Container images older than ninety days are garbage collected from the "
    "mirror every Sunday night by a cron job on the storage host.",
    "Access keys for the artifact storage bucket are rotated once per quarter "
    "by the platform team and handed over through the password manager.",
    "The staging cluster pulls through the same mirror, and its pull-through "
    "cache hides short registry interruptions from the developers.",
    "Certificates for the public API gateway are issued by the corporate CA "
    "and renewed by hand once a year, tracked in the team calendar.",
    "The mirror runs on two hosts behind a virtual address; failing over "
    "between them is a one-line change in the load balancer config.",
    "Docker login credentials for the mirror live in the CI secret store and "
    "are injected into the build container at runtime.",
)

TOP_K = 5
MAX_RESULTS = 10

#: How much better than the legacy whole-node vector the delivered score has to
#: be. Measured 2.44x (0.5393 against 0.2208); the pre-change channel scores
#: exactly 1.00x here by construction, because there the delivered score *is*
#: the whole-node cosine.
MIN_TAIL_GAIN = 1.5

#: Measurement only -- every assertion is in the parent. Reads its inputs as
#: JSON on stdin and writes one JSON object to stdout, so the corpus above is
#: defined exactly once.
PROBE = r"""
import json, sys, tempfile
from pathlib import Path

payload = json.load(sys.stdin)

from living_memory.chunking import (
    ENCODER_SPECIAL_TOKENS,
    MODEL_MAX_SEQ_LENGTH,
    chunk_text,
    get_tokenizer,
)
from living_memory.embeddings import LocalEmbeddingModel, cosine_similarity
from living_memory.retrieval import MemoryRecallService
from living_memory.storage import MemoryStore

content = payload["content"]
query = payload["query"]
scope = payload["scope"]

probe_model = LocalEmbeddingModel()
probe_model.warmup()
if probe_model._sentence_transformer() is None:
    print(json.dumps({"model_available": False}))
    raise SystemExit(0)

tokenizer = get_tokenizer()
spans = tokenizer.token_spans(content)
content_tokens = MODEL_MAX_SEQ_LENGTH - ENCODER_SPECIAL_TOKENS
visible_char_end = len(content) if len(spans) <= content_tokens else spans[content_tokens - 1].end

result = {
    "model_available": True,
    "tokenizer": tokenizer.name,
    "tokens": len(spans),
    "content_chars": len(content),
    "visible_char_end": visible_char_end,
    "chunks": len(chunk_text(content)),
}

with tempfile.TemporaryDirectory(prefix="lm-tail-regression-") as directory:
    with MemoryStore(Path(directory) / "memory.sqlite3") as store:
        tail_id = store.create_node(
            level="trace", content=content, context={"scope": scope, "agent": "regression"}
        ).id
        for text in payload["distractors"]:
            store.create_node(
                level="trace", content=text, context={"scope": scope, "agent": "regression"}
            )
        service = MemoryRecallService(store)
        found = service.memory_recall(query, scope=scope, max_results=payload["max_results"])
        query_vector = service.embedder.embed(query)
        result["tail_id"] = tail_id
        result["ranked"] = [
            {
                "node_id": item.node.id,
                "score": item.score,
                "bm25_score": item.bm25_score,
                "vector_score": item.vector_score,
                "graph_score": item.graph_score,
                "head": item.node.content[:60],
                # What the pre-chunking channel scored: the encoder truncates at
                # 128 tokens, so one vector of the whole node *is* the vector it
                # stored. Recomputed rather than read out of nodes.embedding, so
                # this stays true after an operator drops that column.
                "whole_node_cosine": max(
                    0.0, cosine_similarity(query_vector, probe_model.embed(item.node.content))
                ),
            }
            for item in found
        ]
        result["tail_chunk_rows"] = sum(
            1
            for node_id, _ordinal, _view in store.iter_chunk_embedding_rows(scope=scope)
            if node_id == tail_id
        )

print(json.dumps(result))
"""


def run_probe() -> dict[str, Any]:
    """Run the model-dependent probe in a child process and return its JSON."""

    root = Path(__file__).resolve().parent.parent
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), *([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])]
    )
    # The whole point of the child: the real encoder, with nothing of it left
    # behind in this process.
    environment.pop("LIVING_MEMORY_EMBEDDING_BACKEND", None)
    environment.pop("LIVING_MEMORY_CHUNK_TOKENIZER", None)

    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        input=json.dumps(
            {
                "content": TAIL_CONTENT,
                "query": QUERY,
                "scope": SCOPE,
                "distractors": list(DISTRACTORS),
                "max_results": MAX_RESULTS,
            }
        ),
        capture_output=True,
        text=True,
        env=environment,
        timeout=600,
    )
    assert completed.returncode == 0, f"probe failed:\n{completed.stderr}"
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def probe() -> dict[str, Any]:
    measured = run_probe()
    if not measured.get("model_available"):
        pytest.skip(
            "sentence-transformers model is not in the local cache; this test measures "
            "the encoder's 128-token horizon and is meaningless against the hash fallback"
        )
    if not str(measured.get("tokenizer", "")).startswith("model:"):
        pytest.skip(
            f"tokenizer is {measured.get('tokenizer')!r}, not the model's, so chunk "
            "boundaries would not be the ones the encoder sees"
        )
    return measured


def test_the_answer_really_is_past_the_encoder_horizon(probe: dict[str, Any]) -> None:
    """The fixture's premise, measured with the encoder's own tokenizer."""

    visible_end = probe["visible_char_end"]
    assert visible_end < len(TAIL_HEAD), (
        f"the encoder sees {visible_end} chars, past the {len(TAIL_HEAD)}-char head: "
        "the node is no longer a tail case"
    )
    for term in ANSWER_TERMS:
        assert term in TAIL_ANSWER
        first = TAIL_CONTENT.lower().find(term)
        assert first >= visible_end, f"{term!r} is visible to a single vector at char {first}"
    assert probe["chunks"] > 1, "the chunker cut this node into one window, so there is no max-pool"
    assert probe["tail_chunk_rows"] > 1, "the node was stored as a single chunk"


def test_tail_grounded_node_is_recalled_in_the_top_five(probe: dict[str, Any]) -> None:
    """The house rule: this is the assertion that fails on the pre-change code."""

    ranked = probe["ranked"]
    tail_id = probe["tail_id"]
    ranked_ids = [item["node_id"] for item in ranked]
    assert len(ranked_ids) >= TOP_K, "with fewer than five results a top-5 claim is vacuous"
    assert tail_id in ranked_ids, "the tail node was never returned at all"

    rank = ranked_ids.index(tail_id) + 1
    tail = ranked[rank - 1]
    assert rank <= TOP_K, (
        f"tail node ranked {rank} of {len(ranked_ids)} "
        f"(vector_score {tail['vector_score']:.4f}, "
        f"whole-node cosine {tail['whole_node_cosine']:.4f})"
    )


def test_the_vector_channel_is_what_found_it(probe: dict[str, Any]) -> None:
    """Not bm25, and not a score a single whole-node vector could have produced."""

    tail_id = probe["tail_id"]
    ranked = probe["ranked"]
    tail = next(item for item in ranked if item["node_id"] == tail_id)

    assert tail["bm25_score"] == 0.0, (
        "the query reached the node lexically; it no longer tests the vector channel"
    )
    assert tail["vector_score"] >= MIN_TAIL_GAIN * tail["whole_node_cosine"], (
        f"vector_score {tail['vector_score']:.4f} is not meaningfully above the whole-node "
        f"cosine {tail['whole_node_cosine']:.4f}: the tail is still invisible"
    )

    # The fixture is a genuine tail case rather than a corpus the old channel
    # would also have solved: re-ranking the very same results by that
    # whole-node cosine drops this node out of the top 5.
    legacy_order = sorted(ranked, key=lambda item: item["whole_node_cosine"], reverse=True)
    legacy_rank = [item["node_id"] for item in legacy_order].index(tail_id) + 1
    assert legacy_rank > TOP_K, (
        f"a single whole-node vector already ranks the node {legacy_rank}; "
        "the fixture does not isolate the change"
    )
