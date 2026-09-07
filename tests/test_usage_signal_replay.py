"""Contracts for the shared usage-signal corpus/replay scripts.

``scripts/usage_signal_corpus.py`` extracts closed recall events once;
``scripts/usage_signal_replay.py`` re-grades them under the checkout's
tokenizer and threshold. Every usage-signal vector publishes its before/after
through these two, so their classification rules are pinned here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


corpus = _load("usage_signal_corpus")
replay = _load("usage_signal_replay")


def test_script_class_splits_cyrillic_latin_and_mixed() -> None:
    assert corpus.script_class("alembic migration checksum drift") == "lat"
    assert corpus.script_class("миграция базы данных сломала деплой") == "cyr"
    assert corpus.script_class("deploy failed: миграция alembic checksum drift staging rollout") == "mixed"
    assert corpus.script_class("12345 /// ---") == "lat"
    assert corpus.pair_class("lat", "lat") == "lat/lat"
    assert corpus.pair_class("cyr", "lat") == "cyr-any"
    assert corpus.pair_class("lat", "mixed") == "cyr-any"


def test_relatedness_bands_use_both_cuts() -> None:
    assert replay.relatedness(0.5, related=0.5, unrelated=0.3) == "related"
    assert replay.relatedness(0.3, related=0.5, unrelated=0.3) == "unrelated"
    assert replay.relatedness(0.4, related=0.5, unrelated=0.3) == "middle"


def _record(
    host: str,
    event_id: str,
    trace: str,
    results: list[tuple[str, str, float, list[dict[str, Any]]]],
    *,
    query_cosine: float,
) -> dict[str, Any]:
    return {
        "corpus_version": 1,
        "host": host,
        "event": {"id": event_id, "query": "q", "scope": "project:x", "created_at": "t"},
        "trace": {"id": f"{event_id}-trace", "content": trace, "script": corpus.script_class(trace)},
        "query_trace_cosine": query_cosine,
        "results": [
            {
                "node_id": node_id,
                "rank": rank,
                "content": content,
                "pair_script": corpus.pair_class(
                    corpus.script_class(content), corpus.script_class(trace)
                ),
                "node_trace_cosine": cosine,
                "lookups": lookups,
            }
            for rank, (node_id, content, cosine, lookups) in enumerate(results)
        ],
    }


def test_replay_counts_grounding_lookups_overlap_and_union() -> None:
    used = "redis eviction storm traced to a runaway zset with unbounded members"
    unused = "terraform state lock stuck behind an abandoned dynamodb lease record"
    trace = f"confirmed that {used} and closed it out"
    same = [{"same_transport": True, "lag_seconds": 30.0}]
    foreign = [{"same_transport": False, "lag_seconds": 30.0}]
    late = [{"same_transport": True, "lag_seconds": 10 * 86400.0}]
    records = [
        _record(
            "sfx",
            "e1",
            trace,
            [("n-used", used, 0.9, same), ("n-unused", unused, 0.1, foreign)],
            query_cosine=0.8,
        ),
        _record(
            "sfx",
            "e2",
            "unrelated closing note about a toolbar palette",
            [("n-late", unused, 0.2, late), ("n-fetched", used, 0.2, same)],
            query_cosine=0.1,
        ),
    ]
    report = replay.replay(
        records,
        thresholds=[0.25],
        related=0.5,
        unrelated=0.3,
        lookup_window=86400.0,
        same_transport=True,
    )
    pooled = report["pooled"][0]
    # e1 grounds the used node only; e2 grounds nothing (trace is off-topic).
    assert pooled["pairs"] == {"n": 4, "hit": 1, "share": 0.25}
    assert pooled["closures"] == {"n": 2, "hit": 1, "share": 0.5}
    assert pooled["closures_by_relatedness"]["related"] == {"n": 1, "hit": 1, "share": 1.0}
    assert pooled["closures_by_relatedness"]["unrelated"] == {"n": 1, "hit": 0, "share": 0.0}
    assert pooled["pairs_by_relatedness"]["related"]["hit"] == 1
    assert pooled["pairs_by_relatedness"]["unrelated"]["hit"] == 0
    lookups = pooled["lookups"]
    # Same-transport lookups inside the window: n-used (also grounded) and
    # n-fetched; the cross-transport and the ten-day-late ones do not count.
    assert lookups["pairs"] == {"n": 4, "hit": 2, "share": 0.5}
    assert lookups["overlap_pairs_grounded_and_looked_up"] == 1
    assert lookups["lookup_only_pairs"] == 1
    assert lookups["closures"]["hit"] == 2
    assert lookups["closures_grounded_or_looked_up"] == {"n": 2, "hit": 2, "share": 1.0}
    # Relaxing the transport predicate admits the cross-transport fetch.
    relaxed = replay.replay(
        records, thresholds=[0.25], related=0.5, unrelated=0.3,
        lookup_window=86400.0, same_transport=False,
    )["pooled"][0]["lookups"]
    assert relaxed["pairs"]["hit"] == 3
    assert report["per_host"]["sfx"][0]["pairs"] == pooled["pairs"]


def test_markdown_render_lists_every_threshold(tmp_path: Path) -> None:
    records = [
        _record("alt", "e", "note", [("n", "content", 0.4, [])], query_cosine=0.4)
    ]
    report = replay.replay(
        records, thresholds=[0.1, 0.2], related=0.5, unrelated=0.3,
        lookup_window=0.0, same_transport=True,
    )
    text = replay.render_markdown(report, "title")
    assert "| 0.100 |" in text and "| 0.200 |" in text
    assert "## alt" in text and "## pooled" in text


@pytest.mark.parametrize("bad", ["", "   "])
def test_encoder_cosine_of_blank_text_is_zero(bad: str) -> None:
    class _Model:
        model_name = "stub"

        def embed(self, text: str) -> list[float]:  # pragma: no cover - never reached
            raise AssertionError("blank text must not be embedded")

    assert corpus.Encoder(_Model()).cosine(bad, "anything") == 0.0
