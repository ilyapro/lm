"""Frozen synthetic corpus for recall's max_results/delivery boundary.

Run ``python3 tests/fixtures/recall_distinct_slots.py --baseline`` against the
baseline checkout or ``--candidate`` against the changed checkout. The script
uses a fresh temporary SQLite store and the real MCP handlers. Do not tune
CORPUS or QUERIES after this file's freeze commit; both implementations must
read the same inputs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

# This fixture runs as a script, outside pytest's pythonpath configuration.
# Prefer this checkout over an editable installation in the shared checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from living_memory.chunking import TextChunk
from living_memory.server import create_mcp_server


SCOPE = "project:synthetic-distinct-slots"
STEM = "quillon vantrex ballast survey"
# key, content, pooled-vector angle. The first two state exactly the same
# Tuesday condition. Other close facts deliberately change a material field.
CORPUS = (
    ("survey_a", f"{STEM}: the survey runs at slack water; the harbour office countersigns on Tuesday.", 0),
    ("survey_b", f"{STEM}: at slack water the survey runs; Tuesday is the harbour office countersign day.", 3),
    ("inventory", f"{STEM}: inventory is short; missing crates were logged at the ferry berth.", 70),
    ("monday", "Harbour survey at slack water: the office countersigns on Monday.", 6),
    ("positive", "Beacon audit: pier lamp is enabled after inspection.", 100),
    ("negative", "Beacon audit: pier lamp is disabled after inspection.", 145),
    ("id_alpha", "Pier register: permit PX-17 authorizes the north berth.", 110),
    ("id_beta", "Pier register: permit PX-18 authorizes the north berth.", 113),
    ("old_rule", "Current quay rule: night cargo may depart without a seal.", 120),
    ("new_rule", "Corrected quay rule: night cargo requires a seal before departure.", 123),
)
QUERIES = (
    ("loss", STEM, 2),
    ("reference", STEM, 3),
    ("distinctions", "pier register permit beacon audit corrected quay rule", 10),
)
CORPUS_SHA256 = hashlib.sha256(
    json.dumps((SCOPE, CORPUS, QUERIES), ensure_ascii=False, separators=(",", ":")).encode()
).hexdigest()


class FakeMCP:
    def __init__(self, name: str, instructions: str | None = None) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, func: Any = None, **kwargs: Any) -> Any:
        def register(inner: Any) -> Any:
            self.tools[str(kwargs.get("name") or inner.__name__)] = inner
            return inner
        return register if func is None else register(func)

    def resource(self, *_args: Any, **_kwargs: Any) -> Any:
        return lambda inner: inner

    def prompt(self, func: Any = None, **_kwargs: Any) -> Any:
        return (lambda inner: inner) if func is None else func


def _vector(width: int, angle: int) -> list[float]:
    result = [0.0] * width
    result[0] = math.cos(math.radians(angle))
    result[1] = math.sin(math.radians(angle))
    return result


@contextmanager
def _fixture_environment():
    # Setup recall and measured queries must use the same encoder everywhere.
    # Chunk replacement alone does not freeze the setup recall's access writes.
    clock = ["2099-01-01T00:00:00Z"]
    with patch.dict(os.environ, {"LIVING_MEMORY_EMBEDDING_BACKEND": "hash"}):
        with patch("living_memory.storage._utc_now", side_effect=lambda: clock[0]):
            yield lambda: clock.__setitem__(0, "2099-01-01T00:00:01Z")


def _run(*, expect_baseline: bool) -> dict[str, Any]:
    # Ambient machine settings must not change the frozen comparison.
    for knob in (
        "LM_RECALL_NEAR_DUP_COSINE", "LM_RECALL_NEAR_DUP_LENGTH_RATIO",
        "LM_NEAR_DUP_IDENTIFIER_VETO", "LM_RECALL_REPEAT_GATING",
        "LM_RECALL_REPEAT_DROP_TRAILING_STUBS", "LM_AUTO_CONSOLIDATE_POLICY",
        "LM_DECAY_SWEEP_INTERVAL_SEC", "LM_DELIVERY_SESSION_DEDUP",
        "LM_SCORE_GATE_MIN_SCORE", "LM_RECALL_MIN_SCORE",
    ):
        os.environ.pop(knob, None)
    with _fixture_environment() as advance_clock, tempfile.TemporaryDirectory(
        prefix="lm-distinct-slots-"
    ) as directory:
        mcp = create_mcp_server(Path(directory) / "memory.sqlite3", mcp_factory=FakeMCP)
        store = mcp.memory_store
        ids: dict[str, str] = {}
        for key, content, _angle in CORPUS:
            node = store.create_node(
                level="trace", content=content,
                context={"scope": SCOPE, "agent": "synthetic-fixture", "task": "paired-recall"},
            )
            ids[key] = node.id
        store.create_connection(ids["new_rule"], ids["old_rule"], "supersedes")
        # Keep every delivered node's updated_at distinct from created_at,
        # regardless of which real second the runner happens to cross.
        advance_clock()
        # Normal MCP recall drains the new scope. This is setup, not a measured
        # query; the fixed pooled vectors then replace encoder-specific chunks.
        mcp.tools["memory_recall"](query=STEM, scope=SCOPE, max_results=len(CORPUS))
        width = len(store.list_node_chunks(ids["survey_a"])[0].embedding)
        for key, _content, angle in CORPUS:
            store.replace_node_chunks(ids[key], [(
                TextChunk(text=key, chunk_index=0, token_start=0, token_end=1,
                          char_start=0, char_end=len(key)),
                _vector(width, angle),
            )])
        observations: dict[str, Any] = {}
        for name, query, limit in QUERIES:
            start = time.perf_counter_ns()
            response = mcp.tools["memory_recall"](query=query, scope=SCOPE,
                                                   max_results=limit, depth=0)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            reverse = {value: key for key, value in ids.items()}
            entries = response["results"]
            if name == "distinctions" and not expect_baseline:
                # These dynamic members belong to the measured full wire.
                assert all(
                    "score" in entry
                    and entry["node"]["updated_at"] != entry["node"]["created_at"]
                    for entry in entries
                )
            observations[name] = {
                "query": query, "max_results": limit,
                "keys": [reverse[entry["node"]["id"]] for entry in entries],
                "delivery": [entry["delivery"] for entry in entries],
                "duplicate_of": [reverse.get(entry.get("content_ref", {}).get("duplicate_of"))
                                 for entry in entries],
                "response_bytes": len(json.dumps(response, ensure_ascii=False).encode()),
                "elapsed_ms": round(elapsed_ms, 3),
                "event_result_count": response["count"],
            }
        loss = observations["loss"]
        reference = observations["reference"]
        if expect_baseline:
            assert set(loss["keys"]) == {"survey_a", "survey_b"}, loss
            assert "near_duplicate" in loss["delivery"], loss
            assert "inventory" not in loss["keys"] and "inventory" in reference["keys"], reference
        assert "inventory" in store.get_node(ids["inventory"]).content.lower()
        return {
            "corpus_sha256": CORPUS_SHA256,
            "corpus_count": len(CORPUS),
            "queries": observations,
            "loss_repeat_slots": loss["delivery"].count("near_duplicate"),
            "independent_fact_lost": "inventory" if "inventory" not in loss["keys"] else None,
            # A reader needs the inventory fact. On the baseline it takes a
            # second MCP recall at max_results=3; after selection changes,
            # this same accounting can show whether one answer suffices.
            "necessary_mcp_calls": 1 if "inventory" in loss["keys"] else 2,
            "necessary_response_bytes": loss["response_bytes"] + (
                0 if "inventory" in loss["keys"] else reference["response_bytes"]
            ),
            "necessary_lookup_calls": 0,
            "necessary_lookup_bytes": 0,
            "correction_edge": ["new_rule", "old_rule", "supersedes"],
            "material_distinctions": [
                ["survey_a", "monday", "day"],
                ["positive", "negative", "polarity"],
                ["id_alpha", "id_beta", "permit identifier"],
                ["old_rule", "new_rule", "current correction"],
            ],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    aggregate = _run(expect_baseline=args.baseline)
    assert CORPUS_SHA256 == "64ba90bf6183751e3acd3b3d0739b7b5de96ba795a26ae0f550fb44063ab5742"
    if args.baseline:
        target = Path("artifacts/recall-distinct-slots/baseline.json")
        frozen = json.loads(target.read_text())
        assert frozen["corpus_sha256"] == CORPUS_SHA256
        for name in ("loss", "reference", "distinctions"):
            for field in ("query", "max_results", "keys", "delivery", "duplicate_of"):
                assert aggregate["queries"][name][field] == frozen["queries"][name][field]
    else:
        frozen = json.loads(Path("artifacts/recall-distinct-slots/candidate.json").read_text())
        assert frozen["corpus_sha256"] == aggregate["corpus_sha256"] == CORPUS_SHA256
        assert frozen["corpus_count"] == aggregate["corpus_count"] == len(CORPUS)
        # Runtime jitter is expected. Every other field, including full wire
        # bytes, result order, delivery classes and work accounting, is stable.
        for name, query, limit in QUERIES:
            observed = aggregate["queries"][name]
            recorded = frozen["queries"][name]
            assert observed["query"] == recorded["query"] == query
            assert observed["max_results"] == recorded["max_results"] == limit
            assert {k: v for k, v in observed.items() if k != "elapsed_ms"} == {
                k: v for k, v in recorded.items() if k != "elapsed_ms"
            }, name
            assert observed["elapsed_ms"] >= 0 and recorded["elapsed_ms"] >= 0
            assert observed["event_result_count"] == len(observed["keys"])
        for field in (
            "loss_repeat_slots", "independent_fact_lost", "necessary_mcp_calls",
            "necessary_response_bytes", "necessary_lookup_calls", "necessary_lookup_bytes",
            "correction_edge", "material_distinctions",
        ):
            assert aggregate[field] == frozen[field], field
        loss = aggregate["queries"]["loss"]
        reference = aggregate["queries"]["reference"]
        distinctions = aggregate["queries"]["distinctions"]
        assert "survey_a" in loss["keys"]
        assert {"survey_a", "survey_b", "inventory"} <= set(reference["keys"])
        assert {"new_rule", "positive", "negative", "id_alpha", "id_beta"} <= set(
            distinctions["keys"]
        )
        assert distinctions["keys"][0] == "new_rule" and "old_rule" not in distinctions["keys"]
        assert all(delivery == "full" for delivery in distinctions["delivery"])
        assert aggregate["correction_edge"] == ["new_rule", "old_rule", "supersedes"]
        assert aggregate["necessary_mcp_calls"] == (1 if "inventory" in loss["keys"] else 2)
        assert aggregate["necessary_response_bytes"] == loss["response_bytes"] + (
            0 if "inventory" in loss["keys"] else reference["response_bytes"]
        )
        assert aggregate["loss_repeat_slots"] == loss["delivery"].count("near_duplicate")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
