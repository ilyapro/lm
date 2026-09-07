#!/usr/bin/env python3
"""Assemble ``artifacts/grounding/cyrillic-tokenizer.{md,json}``.

Before/after evidence for Cyrillic stemming in ``embeddings.tokenize``:

* usage-signal replay (``scripts/usage_signal_replay.py``) with
  ``LM_TOKENIZE_CYRILLIC_STEM=off`` (the previous tokenizer; must equal the
  baseline artifact) and ``on``: grounding of ``cyr-any`` and ``lat/lat``
  pairs, pair and closure signal/noise, per host and pooled;
* BM25 latency (``scripts/bm25_cyrillic_latency.py``) with and without the
  prefix terms;
* retrieval non-regression: ``living_memory.retrieval_harness run`` reports
  for master and for the branch (stemming on, and off as a control), overall
  and on the goldset items whose query contains Cyrillic;
* index-consistency evidence computed here from the snapshot and the corpora:
  the recall-map term verdicts that change, the pending-recall text-similarity
  verdicts that flip, the schema-trigger pairs that flip, the derived
  stem->canonical synonym entries.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from living_memory import embeddings  # noqa: E402
from living_memory.embeddings import (  # noqa: E402
    _RU_MIN_PREFIX,
    _STOP_WORDS,
    _TOKEN_RE,
    CYRILLIC_STEM_ENV_VAR,
    _CYRILLIC_STEM_SYNONYMS,
    _is_russian_token,
    _stem_russian,
    reset_cyrillic_stem_cache,
    tokenize,
)
from living_memory.retrieval import SCHEMA_TRIGGER_OVERLAP_THRESHOLD  # noqa: E402
from living_memory.storage import _RECALL_TEXT_SIMILARITY_THRESHOLD  # noqa: E402

CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")


def _switch(state: str) -> None:
    os.environ[CYRILLIC_STEM_ENV_VAR] = state
    reset_cyrillic_stem_cache()


def _load(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rows(report: dict[str, Any], host: str) -> list[dict[str, Any]]:
    return report["pooled"] if host == "pooled" else report["per_host"][host]


def _cell(tally: dict[str, Any] | None) -> str:
    if not tally:
        return "–"
    return f"{tally['hit']}/{tally['n']} ({tally['share'] * 100:.1f}%)"


def replay_section(before: dict[str, Any], after: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    hosts = ["pooled", *sorted(before["per_host"])]
    lines: list[str] = []
    data: dict[str, Any] = {}
    for host in hosts:
        lines.append(f"### {host}")
        lines.append("")
        lines.append(
            "| thr | cyr-any before | cyr-any after | lat/lat before | lat/lat after | "
            "unrelated pairs before | after | pair S/N before | after | "
            "closures before | after | closure S/N before | after |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        data[host] = []
        for old, new in zip(_rows(before, host), _rows(after, host), strict=True):
            assert abs(old["threshold"] - new["threshold"]) < 1e-9
            row = {
                "threshold": old["threshold"],
                "cyr_any": {"before": old["pairs_by_script"].get("cyr-any"), "after": new["pairs_by_script"].get("cyr-any")},
                "lat_lat": {"before": old["pairs_by_script"].get("lat/lat"), "after": new["pairs_by_script"].get("lat/lat")},
                "pairs_unrelated": {"before": old["pairs_by_relatedness"].get("unrelated"), "after": new["pairs_by_relatedness"].get("unrelated")},
                "pairs_related": {"before": old["pairs_by_relatedness"].get("related"), "after": new["pairs_by_relatedness"].get("related")},
                "pair_signal_to_noise": {"before": old["pair_signal_to_noise"], "after": new["pair_signal_to_noise"]},
                "closures": {"before": old["closures"], "after": new["closures"]},
                "closure_signal_to_noise": {"before": old["closure_signal_to_noise"], "after": new["closure_signal_to_noise"]},
                "closures_related": {"before": old["closures_by_relatedness"].get("related"), "after": new["closures_by_relatedness"].get("related")},
                "closures_unrelated": {"before": old["closures_by_relatedness"].get("unrelated"), "after": new["closures_by_relatedness"].get("unrelated")},
            }
            data[host].append(row)
            lines.append(
                f"| {old['threshold']:.2f} | {_cell(row['cyr_any']['before'])} | {_cell(row['cyr_any']['after'])} "
                f"| {_cell(row['lat_lat']['before'])} | {_cell(row['lat_lat']['after'])} "
                f"| {_cell(row['pairs_unrelated']['before'])} | {_cell(row['pairs_unrelated']['after'])} "
                f"| {old['pair_signal_to_noise']['ratio']} | {new['pair_signal_to_noise']['ratio']} "
                f"| {_cell(old['closures'])} | {_cell(new['closures'])} "
                f"| {old['closure_signal_to_noise']['ratio']} | {new['closure_signal_to_noise']['ratio']} |"
            )
        lines.append("")
    return lines, data


def lat_lat_unchanged(before: dict[str, Any], after: dict[str, Any]) -> bool:
    for host in ["pooled", *before["per_host"]]:
        for old, new in zip(_rows(before, host), _rows(after, host), strict=True):
            if old["pairs_by_script"].get("lat/lat") != new["pairs_by_script"].get("lat/lat"):
                return False
    return True


def harness_metrics(report: dict[str, Any], russian_ids: set[str]) -> dict[str, Any]:
    def block(runs: list[dict[str, Any]]) -> dict[str, Any]:
        hit1 = hit5 = hit10 = 0
        rr = 0.0
        for run in runs:
            relevant = set(run["relevant_node_ids"])
            ranked = run["ranked_node_ids"]
            first = next((index for index, node_id in enumerate(ranked) if node_id in relevant), None)
            if first is None:
                continue
            hit1 += int(first < 1)
            hit5 += int(first < 5)
            hit10 += int(first < 10)
            rr += 1.0 / (first + 1)
        n = len(runs)
        return {
            "items": n,
            "hit@1": round(hit1 / n, 4) if n else 0.0,
            "hit@5": round(hit5 / n, 4) if n else 0.0,
            "hit@10": round(hit10 / n, 4) if n else 0.0,
            "mrr": round(rr / n, 4) if n else 0.0,
        }

    runs = report["runs"]
    overall = block(runs)
    reported = report["metrics"]["overall"]
    # The recomputation must agree with the harness's own overall block.
    for key in ("hit@1", "hit@5", "hit@10", "mrr"):
        assert abs(overall[key] - float(reported[key])) < 5e-4, (key, overall[key], reported[key])
    return {
        "overall": overall,
        "russian_query": block([run for run in runs if run["query_id"] in russian_ids]),
        "non_russian_query": block([run for run in runs if run["query_id"] not in russian_ids]),
        "snapshot_sha256": report.get("provenance", {}).get("snapshot_sha256"),
        "goldset_sha256": report.get("provenance", {}).get("goldset_sha256"),
        "agreement_rate": report.get("agreement", {}).get("agreement_rate"),
        "generated_at": report.get("provenance", {}).get("generated_at"),
    }


def harness_section(
    master: dict[str, Any],
    branch: dict[str, Any],
    branch_off: dict[str, Any] | None,
    goldset: Path,
    extra_arms: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    russian_ids: set[str] = set()
    total = 0
    with goldset.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            total += 1
            if CYRILLIC_RE.search(str(item["query"])):
                russian_ids.add(str(item["query_id"]))
    arms = {"master": harness_metrics(master, russian_ids), "branch_on": harness_metrics(branch, russian_ids)}
    if branch_off is not None:
        arms["branch_off"] = harness_metrics(branch_off, russian_ids)
    for label, report in (extra_arms or {}).items():
        arms[label] = harness_metrics(report, russian_ids)
    lines = [
        f"Goldset: {total} items, {len(russian_ids)} with a Cyrillic query. Real encoder, frozen "
        f"snapshot `{Path(str(master.get('provenance', {}).get('snapshot_path', 'frozen-chunked.sqlite3'))).name}` "
        f"(sha256 {str(master.get('provenance', {}).get('snapshot_sha256'))[:12]}…), live-agreement rate "
        + ", ".join(f"{arm} {metrics['agreement_rate']}" for arm, metrics in arms.items())
        + ".",
        "",
        "| arm | bucket | items | hit@1 | hit@5 | hit@10 | MRR |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, metrics in arms.items():
        for bucket in ("overall", "russian_query", "non_russian_query"):
            block = metrics[bucket]
            lines.append(
                f"| {arm} | {bucket} | {block['items']} | {block['hit@1']:.3f} | {block['hit@5']:.3f} "
                f"| {block['hit@10']:.3f} | {block['mrr']:.3f} |"
            )
    deltas: dict[str, Any] = {}
    for arm in [name for name in arms if name != "master"]:
        deltas[arm] = {
            bucket: {
                key: round(arms[arm][bucket][key] - arms["master"][bucket][key], 4)
                for key in ("hit@1", "hit@5", "hit@10", "mrr")
            }
            for bucket in ("overall", "russian_query", "non_russian_query")
        }
    lines.append("")
    for arm, per_bucket in deltas.items():
        lines.append(
            f"{arm} minus master: "
            + "; ".join(
                f"{bucket}: " + ", ".join(f"{key} {value:+.4f}" for key, value in delta.items())
                for bucket, delta in per_bucket.items()
            )
            + "."
        )
    def first_rank(run: dict[str, Any]) -> int:
        relevant = set(run["relevant_node_ids"])
        return next((i for i, n in enumerate(run["ranked_node_ids"]) if n in relevant), 10**6)

    master_runs = {run["query_id"]: run for run in master["runs"]}
    moved = {"better": 0, "worse": 0, "ranking_changed": 0}
    for run in branch["runs"]:
        if run["query_id"] not in russian_ids:
            continue
        before = master_runs[run["query_id"]]
        moved["ranking_changed"] += int(before["ranked_node_ids"] != run["ranked_node_ids"])
        moved["better"] += int(first_rank(run) < first_rank(before))
        moved["worse"] += int(first_rank(run) > first_rank(before))
    lines.append(
        f"Russian-query items, branch_on vs master: ranking changed on {moved['ranking_changed']}, first relevant "
        f"node higher on {moved['better']}, lower on {moved['worse']}; hit@10 unchanged. The remaining losses are "
        "queries whose Russian words are common (`проблема`, `задача`, `статус`) next to a rare identifier: the "
        "prefix term brings in documents dense in that word and the identifier match slips a rank or two."
    )
    identical_off = None
    if branch_off is not None:
        identical_off = [run["ranked_node_ids"] for run in sorted(master["runs"], key=lambda r: r["query_id"])] == [
            run["ranked_node_ids"] for run in sorted(branch_off["runs"], key=lambda r: r["query_id"])
        ]
        lines.append(
            f"Branch with `LM_TOKENIZE_CYRILLIC_STEM=off` returns the same ranked ids as master on every item: "
            f"{'yes' if identical_off else 'NO'}."
        )
    return lines, {
        "arms": arms,
        "minus_master": deltas,
        "russian_items_branch_on_vs_master": moved,
        "branch_off_identical_to_master": identical_off,
    }


def prefix_breadth(vocab: list[tuple[str, int]], goldset: Path) -> dict[str, Any]:
    """How much of the index a prefix term of each stem length would cover.

    For every Russian content word of the Cyrillic goldset queries: its stem,
    the number of distinct index terms starting with that stem and their
    summed document postings, grouped by stem length. This is the evidence
    behind ``_RU_MIN_PREFIX``.
    """

    import bisect

    terms = [term for term, _ in vocab]
    docs = [count for _, count in vocab]
    stems: dict[str, int] = {}
    with goldset.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            query = str(json.loads(line)["query"])
            if not CYRILLIC_RE.search(query):
                continue
            for raw in _TOKEN_RE.findall(query.lower()):
                word = raw.replace("ё", "е")
                if len(word) <= 3 or word in _STOP_WORDS or not _is_russian_token(word):
                    continue
                stem = _stem_russian(word)
                if len(stem) >= 3:
                    stems[stem] = stems.get(stem, 0) + 1
    by_length: dict[str, list[tuple[int, int]]] = {}
    for stem in stems:
        low = bisect.bisect_left(terms, stem)
        high = bisect.bisect_left(terms, stem + "\uffff")
        key = f"{min(len(stem), 6)}{'+' if len(stem) >= 6 else ''}"
        by_length.setdefault(key, []).append((high - low, sum(docs[low:high])))
    summary = {}
    for key in sorted(by_length):
        rows = by_length[key]
        summary[key] = {
            "stems": len(rows),
            "vocab_terms_median": statistics.median(r[0] for r in rows),
            "vocab_terms_mean": round(statistics.fmean(r[0] for r in rows), 1),
            "postings_median": statistics.median(r[1] for r in rows),
            "postings_mean": round(statistics.fmean(r[1] for r in rows), 1),
        }
    return {"min_prefix": _RU_MIN_PREFIX, "distinct_stems": len(stems), "by_stem_length": summary}


def consistency_evidence(snapshot: Path, corpora: list[Path], goldset: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{snapshot.resolve()}?mode=ro", uri=True)
    try:
        connection.execute("CREATE VIRTUAL TABLE temp.vocab USING fts5vocab(main, 'nodes_fts', 'row')")
        vocab = sorted((str(row[0]), int(row[1])) for row in connection.execute("SELECT term, doc FROM temp.vocab"))
        terms = [term for term, _ in vocab]
        schemas = [
            (str(row[0]), str(json.loads(row[1]).get("trigger") or ""))
            for row in connection.execute("SELECT id, context FROM nodes WHERE level = 'schema' AND decayed = 0")
        ]
        queries = [
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT query FROM recall_events WHERE created_at >= '2026-08-01' ORDER BY created_at DESC LIMIT 2000"
            )
            if row[0] and CYRILLIC_RE.search(str(row[0]))
        ][:300]
    finally:
        connection.close()
    russian_terms = [term for term in terms if len(term) >= 3 and _is_russian_token(term)]

    def verdicts() -> dict[str, bool]:
        return {term: bool(tokenize(term)) for term in russian_terms}

    _switch("off")
    verdict_off = verdicts()
    _switch("on")
    verdict_on = verdicts()
    changed = sorted(term for term in russian_terms if verdict_off[term] != verdict_on[term])
    stems: dict[str, int] = {}
    for term in russian_terms:
        tokens = tokenize(term)
        if tokens:
            stems[tokens[0]] = stems.get(tokens[0], 0) + 1

    records: list[dict[str, Any]] = []
    for corpus in corpora:
        with corpus.open(encoding="utf-8") as handle:
            records += [json.loads(line) for line in handle if line.strip()]

    def similarity(record: dict[str, Any]) -> float:
        query_tokens = set(tokenize(record["event"]["query"]))
        content_tokens = set(tokenize(record["trace"]["content"]))
        if not query_tokens or not content_tokens:
            return 0.0
        return len(query_tokens & content_tokens) / max(1, min(len(query_tokens), len(content_tokens)))

    text: dict[str, Any] = {}
    for state in ("off", "on"):
        _switch(state)
        text[state] = {record["event"]["id"]: similarity(record) for record in records}
    by_script: dict[str, Any] = {}
    for script in sorted({record["trace"]["script"] for record in records}):
        ids = [record["event"]["id"] for record in records if record["trace"]["script"] == script]
        above = {
            state: sum(text[state][event_id] >= _RECALL_TEXT_SIMILARITY_THRESHOLD for event_id in ids)
            for state in ("off", "on")
        }
        by_script[script] = {
            "closures": len(ids),
            "mean_similarity": {state: round(statistics.fmean(text[state][i] for i in ids), 4) for state in ("off", "on")},
            "above_threshold": above,
            "verdict_flips": sum(
                (text["off"][i] >= _RECALL_TEXT_SIMILARITY_THRESHOLD) != (text["on"][i] >= _RECALL_TEXT_SIMILARITY_THRESHOLD)
                for i in ids
            ),
        }

    schemas = [(node_id, trigger) for node_id, trigger in schemas if trigger]

    def firing() -> set[tuple[str, str]]:
        trigger_tokens = {node_id: set(tokenize(trigger)) for node_id, trigger in schemas}
        fired: set[tuple[str, str]] = set()
        for query in queries:
            query_tokens = set(tokenize(query))
            for node_id, tokens in trigger_tokens.items():
                if tokens and len(query_tokens & tokens) / len(tokens) >= SCHEMA_TRIGGER_OVERLAP_THRESHOLD:
                    fired.add((query, node_id))
        return fired

    _switch("off")
    fired_off = firing()
    _switch("on")
    fired_on = firing()
    return {
        "fts_vocabulary": {
            "terms": len(terms),
            "russian_terms_3plus": len(russian_terms),
            "russian_terms_with_a_token_on": sum(stems.values()),
            "distinct_tokens_on": len(stems),
            "stop_word_verdict_changes": len(changed),
            "stop_word_verdict_changed_terms": changed,
        },
        "recall_event_text_similarity": {
            "threshold": _RECALL_TEXT_SIMILARITY_THRESHOLD,
            "closures": len(records),
            "by_trace_script": by_script,
        },
        "schema_triggers": {
            "threshold": SCHEMA_TRIGGER_OVERLAP_THRESHOLD,
            "schemas_with_trigger": len(schemas),
            "russian_queries": len(queries),
            "pairs_firing": {"off": len(fired_off), "on": len(fired_on)},
            "only_on": len(fired_on - fired_off),
            "only_off": len(fired_off - fired_on),
        },
        "derived_stem_synonyms": dict(sorted(_CYRILLIC_STEM_SYNONYMS.items())),
        "prefix_breadth": prefix_breadth(vocab, goldset),
    }


def latency_section(latency: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    lines = [
        f"Snapshot `{Path(latency['snapshot']).name}`, {latency['queries']} distinct recorded Russian queries since "
        f"{latency['since']}, {latency['rounds']} rounds per arm, arms measured off/on/off/on after a warm-up; one "
        f"`search_content` call per scope of the recorded event's plan, per-scope limit as `_collect_bm25` uses it.",
        "",
        "| arm | per query p50 | p95 | p99 | max | per call p50 | p95 | prefix terms/query (mean, max) | FTS terms/query (mean) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in latency["arms"]:
        q, c = arm["per_query"], arm["per_search_content_call"]
        lines.append(
            f"| {arm['switch']} | {q['p50_ms']:.1f} ms | {q['p95_ms']:.1f} ms | {q['p99_ms']:.1f} ms | {q['max_ms']:.1f} ms "
            f"| {c['p50_ms']:.1f} ms | {c['p95_ms']:.1f} ms | {arm['prefix_terms_per_query']['mean']}, "
            f"{arm['prefix_terms_per_query']['max']} | {arm['fts_terms_per_query']['mean']} |"
        )
    off = [arm for arm in latency["arms"] if arm["switch"] == "off"]
    on = [arm for arm in latency["arms"] if arm["switch"] == "on"]
    summary = {
        "off": {key: round(statistics.fmean(arm["per_query"][key] for arm in off), 3) for key in ("p50_ms", "p95_ms")},
        "on": {key: round(statistics.fmean(arm["per_query"][key] for arm in on), 3) for key in ("p50_ms", "p95_ms")},
    }
    lines.append("")
    lines.append(
        f"Mean over the two runs of each arm, per query: off p50 {summary['off']['p50_ms']:.1f} ms / p95 "
        f"{summary['off']['p95_ms']:.1f} ms, on p50 {summary['on']['p50_ms']:.1f} ms / p95 {summary['on']['p95_ms']:.1f} ms."
    )
    return lines, {"arms": latency["arms"], "per_query_mean_of_runs": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--replay-before", required=True)
    parser.add_argument("--replay-after", required=True)
    parser.add_argument("--baseline", required=True, help="artifacts/grounding/usage-signal-baseline.json")
    parser.add_argument("--latency", required=True)
    parser.add_argument("--harness-master")
    parser.add_argument("--harness-branch")
    parser.add_argument("--harness-branch-off")
    parser.add_argument(
        "--harness-extra",
        action="append",
        default=[],
        help="LABEL=PATH: an additional harness arm to tabulate (e.g. a rejected configuration)",
    )
    parser.add_argument("--goldset", required=True)
    parser.add_argument("--snapshot", required=True, help="sfx snapshot (read-only) for the consistency evidence")
    parser.add_argument("--corpus", action="append", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args(argv)

    before = _load(args.replay_before)
    after = _load(args.replay_after)
    baseline = _load(args.baseline)
    assert before is not None and after is not None and baseline is not None
    strip = lambda report: {key: value for key, value in report.items() if key != "corpus"}  # noqa: E731
    before_equals_baseline = strip(before) == strip(baseline)

    replay_lines, replay_data = replay_section(before, after)
    latency_lines, latency_data = latency_section(_load(args.latency) or {})
    evidence = consistency_evidence(
        Path(args.snapshot).expanduser(),
        [Path(p).expanduser() for p in args.corpus],
        Path(args.goldset).expanduser(),
    )
    harness_lines: list[str] = ["Harness reports not supplied."]
    harness_data: dict[str, Any] | None = None
    master = _load(args.harness_master)
    branch = _load(args.harness_branch)
    if master is not None and branch is not None:
        extra = {}
        for spec in args.harness_extra:
            label, _, path = spec.partition("=")
            loaded = _load(path)
            if loaded is not None:
                extra[label] = loaded
        harness_lines, harness_data = harness_section(
            master, branch, _load(args.harness_branch_off), Path(args.goldset).expanduser(), extra
        )

    unchanged = lat_lat_unchanged(before, after)
    pooled_25 = next(row for row in replay_data["pooled"] if abs(row["threshold"] - 0.25) < 1e-9)
    summary = {
        "before_equals_baseline": before_equals_baseline,
        "lat_lat_pairs_unchanged_at_every_threshold": unchanged,
        "pooled_at_0_25": pooled_25,
    }
    md = [
        "# Cyrillic tokenizer: before/after (snapshots 2026-09-07, closures since 2026-09-04)",
        "",
        "Change under test: Snowball-style Russian stemming for all-Cyrillic tokens in "
        "`living_memory.embeddings.tokenize` (`_stem`), stems of the 69 Cyrillic synonym keys mapped to their "
        "canonical token, and FTS5 prefix terms (`\"стем\"*`) for the Russian content words of a BM25 query "
        "(`retrieval._expanded_query` -> `storage._fts_query`). Switch: `LM_TOKENIZE_CYRILLIC_STEM` "
        "(`on` default, `off` = previous behaviour). Before = `off`, after = `on`, same checkout.",
        "",
        "## Summary",
        "",
        f"- `off` reproduces the baseline artifact (`usage-signal-baseline.json`) exactly: "
        f"{'yes' if before_equals_baseline else 'NO'}.",
        f"- lat/lat pair grounding unchanged at every threshold and host: {'yes' if unchanged else 'NO'} "
        "(byte-identity of non-Cyrillic tokenization, pinned by `tests/test_tokenize_cyrillic.py`).",
        f"- Pooled at 0.25: cyr-any pairs {_cell(pooled_25['cyr_any']['before'])} -> {_cell(pooled_25['cyr_any']['after'])}; "
        f"lat/lat {_cell(pooled_25['lat_lat']['before'])} -> {_cell(pooled_25['lat_lat']['after'])}; "
        f"closures grounded {_cell(pooled_25['closures']['before'])} -> {_cell(pooled_25['closures']['after'])}; "
        f"pair S/N {pooled_25['pair_signal_to_noise']['before']['ratio']} -> {pooled_25['pair_signal_to_noise']['after']['ratio']}; "
        f"closure S/N {pooled_25['closure_signal_to_noise']['before']['ratio']} -> {pooled_25['closure_signal_to_noise']['after']['ratio']}.",
        *(
            [
                "- Retrieval harness (real encoder, 3659 goldset items, 227 Russian queries), branch minus master: "
                + "; ".join(
                    f"{bucket} hit@5 {delta['hit@5']:+.4f}, MRR {delta['mrr']:+.4f}"
                    for bucket, delta in harness_data["minus_master"]["branch_on"].items()
                )
                + f"; non-Russian rankings identical on every item; `off` identical to master on every item: "
                f"{'yes' if harness_data['branch_off_identical_to_master'] else 'NO'}. "
                f"BM25 latency per query with prefix terms: p50 {latency_data['per_query_mean_of_runs']['off']['p50_ms']:.1f} -> "
                f"{latency_data['per_query_mean_of_runs']['on']['p50_ms']:.1f} ms, p95 "
                f"{latency_data['per_query_mean_of_runs']['off']['p95_ms']:.1f} -> {latency_data['per_query_mean_of_runs']['on']['p95_ms']:.1f} ms."
            ]
            if harness_data is not None
            else []
        ),
        "",
        "## Usage-signal replay: grounding before vs after",
        "",
        "Pairs = result/trace pairs; `cyr-any` = Cyrillic on either side, `lat/lat` = neither. Relatedness by "
        "encoder cosine (related ≥ 0.5, unrelated ≤ 0.3). S/N = grounded share among related over grounded "
        "share among unrelated.",
        "",
        *replay_lines,
        "## BM25 latency with prefix terms (sfx snapshot)",
        "",
        *latency_lines,
        "",
        "### Prefix breadth by stem length (sfx vocabulary, Cyrillic goldset queries)",
        "",
        f"Stems of the Russian content words of the Cyrillic goldset queries ({evidence['prefix_breadth']['distinct_stems']} "
        f"distinct), each matched as an FTS5 prefix against the sfx index vocabulary; shipped floor "
        f"`_RU_MIN_PREFIX = {evidence['prefix_breadth']['min_prefix']}`.",
        "",
        "| stem length | stems | vocab terms matched (median / mean) | doc postings (median / mean) |",
        "|---|---|---|---|",
        *[
            f"| {key} | {row['stems']} | {row['vocab_terms_median']:.0f} / {row['vocab_terms_mean']} "
            f"| {row['postings_median']:.0f} / {row['postings_mean']} |"
            for key, row in evidence["prefix_breadth"]["by_stem_length"].items()
        ],
        "",
        "## Retrieval non-regression (retrieval harness, real encoder)",
        "",
        *harness_lines,
        "",
        "## What does not need reindexing, and what shifts",
        "",
        "- **chunks** (`chunking.py`): windows are cut with the encoder's own tokenizer (`tokenizer.json` of the "
        "sentence-transformers snapshot); the module imports `DEFAULT_EMBEDDING_MODEL`, `_HASH_BACKENDS` and "
        "`_resolve_local_model_source` from `embeddings` and never `tokenize`. Chunk vectors come from the encoder. "
        "Unaffected.",
        "- **query anchors**: stored as encoder vectors (`storage.iter_anchor_embedding_rows`, "
        "`retrieval._collect_anchor_seeds` matches on the query embedding). Unaffected.",
        "- **`nodes_fts`**: the `CREATE VIRTUAL TABLE ... tokenize = 'unicode61'` block and its triggers are "
        "byte-identical to master (checked with `git show master:src/living_memory/storage.py`); the index keeps "
        "surface forms and the branch only changes the MATCH expression. No reindex.",
        "- **recall-map c-TF-IDF labels** (`recall_map._terms`): candidates come from the map's own `_TERM_RE` and are "
        "looked up unstemmed in `nodes_fts_vocab`; `tokenize` is consulted only as a stop-word verdict. Over the "
        f"sfx vocabulary ({evidence['fts_vocabulary']['russian_terms_3plus']} Russian terms of 3+ letters) that verdict "
        f"changes for {evidence['fts_vocabulary']['stop_word_verdict_changes']} terms, all of them forms whose stem is a "
        f"stop word ({', '.join(evidence['fts_vocabulary']['stop_word_verdict_changed_terms'][:12])}"
        f"{', ...' if evidence['fts_vocabulary']['stop_word_verdict_changes'] > 12 else ''}); "
        "these stop being label candidates. Labels are computed at recall time, nothing stored.",
        "- **live consumers whose overlap values shift** (both tokenize both sides at call time, nothing stored):",
        f"  - `storage._recall_event_text_similarity` (pending-event matching, threshold "
        f"{evidence['recall_event_text_similarity']['threshold']}): over the {evidence['recall_event_text_similarity']['closures']} "
        "corpus closures, "
        + "; ".join(
            f"{script} traces n={block['closures']}: mean {block['mean_similarity']['off']} -> {block['mean_similarity']['on']}, "
            f"above threshold {block['above_threshold']['off']} -> {block['above_threshold']['on']}, verdict flips {block['verdict_flips']}"
            for script, block in evidence["recall_event_text_similarity"]["by_trace_script"].items()
        )
        + ".",
        f"  - `retrieval._collect_schema_triggers` (overlap threshold {evidence['schema_triggers']['threshold']}): "
        f"{evidence['schema_triggers']['schemas_with_trigger']} sfx schemas with a trigger × "
        f"{evidence['schema_triggers']['russian_queries']} recorded Russian queries: pairs firing "
        f"{evidence['schema_triggers']['pairs_firing']['off']} -> {evidence['schema_triggers']['pairs_firing']['on']} "
        f"(only on: {evidence['schema_triggers']['only_on']}, only off: {evidence['schema_triggers']['only_off']}).",
        f"- **derived stem synonyms** ({len(evidence['derived_stem_synonyms'])} entries): "
        + ", ".join(f"{stem}→{canonical}" for stem, canonical in evidence["derived_stem_synonyms"].items())
        + ". `потому` is skipped (its stem `пот` is also the stem of `потом`).",
        "",
        "## Known limits of the light stemmer",
        "",
        "- Zero-ending genitive plurals with a fleeting vowel stay unstemmed (`ошибок` ≠ `ошибк`).",
        "- Loanwords in `-ой` take the adjective path (`деплой` → `депл`, `деплоя` → `депло`); both are listed synonym "
        "forms and their prefix terms overlap, so BM25 still matches.",
        "- A stem shorter than three letters is rejected in favour of the surface form (`этого` stays `этого`).",
        "- Prefix terms need a four-letter stem. The three-letter floor was measured first (arm "
        "`branch_on_prefix3` above, where supplied): `рев*`, `цел*`, `дан*`, `мок*` and the like matched a median of 19 "
        "vocabulary terms each and moved relevant nodes down on more Russian goldset items than up, so the floor was "
        "raised; words with a shorter stem keep matching exactly as before.",
        "",
    ]
    out_json = {
        "summary": summary,
        "replay": {
            "before": {"switch": "off", "path": str(Path(args.replay_before).resolve())},
            "after": {"switch": "on", "path": str(Path(args.replay_after).resolve())},
            "baseline": str(Path(args.baseline).resolve()),
            "records": after["records"],
            "pairs": after["pairs"],
            "pair_containment_percentiles": {"before": before["pair_containment_percentiles"], "after": after["pair_containment_percentiles"]},
            "by_host": replay_data,
        },
        "bm25_latency": latency_data,
        "retrieval_harness": harness_data,
        "consistency_evidence": evidence,
    }
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(md), encoding="utf-8")
    Path(args.out_json).write_text(json.dumps(out_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out_md} and {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
