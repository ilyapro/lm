#!/usr/bin/env python3
"""Verify every invariant of the de-identified replay corpus against staging.

Checks (all must pass; exit 0 iff clean):

 1. schema         — every corpus record carries exactly the documented fields
                     in the documented order; re-serializing every parsed
                     record reproduces its file line byte-for-byte.
 2. kept_real      — ids, scopes, timestamps, scores, flags, results arrays,
                     stats equal the staging originals byte-for-byte.
 3. char_counts    — every ``*_chars`` equals the staging original's length
                     (delivery char semantics for context values).
 4. surrogates     — every surrogate has the original's exact length,
                     whitespace (space/tab/newline/CR) verbatim at original
                     positions, all other chars in [a-z] (purity
                     ``^[a-z \\t\\n\\r]*$``).
 5. equality       — global equality classes: identical originals map to one
                     surrogate, distinct originals never share one (checked
                     jointly across dev/eval/holdout).
 6. classes        — class organic|automatic and template_id recompute from
                     the staging originals via the pinned METADATA predicates.
 7. splits         — recompute the pre-declared hash rule from splits.json
                     constants; ids disjoint and exhaustive over the staged
                     export; all local events in holdout; per-split counts and
                     file sha256 match splits.json.
 8. containment    — each split file embeds every node its events reference.
 9. holdout        — holdout carries >=2 distinct non-animal-planet project
                     scopes (local-DB workloads).
10. family         — correction/supersedes-bearing events present in both dev
                     and eval (recomputed, equal to recorded flags).
11. transcripts    — transcript_matched/chars equal transcripts/matched.jsonl.
12. ngram_corpus   — sampled >=12-char n-grams of staged private text are
                     searched in the corpus jsonl files.  A hit is
                     adjudicated positionally against a sentinel copy of the
                     file (surrogate letters replaced by a sentinel char):
                     windows overlapping surrogate spans by <= 3 characters
                     are structural coincidences (JSON syntax + kept-real key
                     names + <=3 pseudo-random letters that happened to match,
                     p <= 26^-1..3 per site); >= 4 overlapping characters, or
                     any unexplained full-window hit, is a leak.  A residual
                     >= 12-char leak below that bar would need >= 9 characters
                     carried by non-surrogate material, which checks 1-11
                     audit field-by-field.
13. ngram_docs     — EVERY 24-gram of the authored tracked files (splits.json,
                     POLICY.md, 02-transform.md, transform/*.py) is searched
                     against the full staged private text; a hit is a leak
                     unless every alphanumeric token in it is a fragment of
                     kept-real-by-design vocabulary (record field/key names,
                     kept-real leaf values such as scopes/ids/timestamps/
                     template fingerprints) or a digit/timestamp fragment.

Touches holdout.jsonl only mechanically (parsing + invariant recomputation);
never prints any original or surrogate text — ids, field names and counts
only.  Full leak diagnostics, when any, go to the PRIVATE staging area
(deid_map/ngram_leaks.jsonl), never to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deid import (  # noqa: E402
    HEX32_RE,
    KEEP_WS,
    PURITY_RE,
    TEMPORAL_HINT_RE,
    classify_class,
    classify_template,
    context_chars_map,
    keeps_real,
    provenance_shape_of,
)
from build_corpus import LOCAL_SLUGS, parse_json_col, read_jsonl  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_STAGING = Path(os.environ.get("AP_STAGING", "/home/sfx/.cache/ap-audit/staging"))
DEFAULT_CORPUS = HERE.parent.parent / "corpus"

EVENT_FIELDS = [
    "type", "id", "source", "created_at", "scope", "requested_scope",
    "resolved_scopes", "depth", "max_results", "class", "template_id",
    "query_surrogate", "query_chars", "agent_surrogate", "agent_chars",
    "task_surrogate", "task_chars", "session_id_surrogate", "session_id_chars",
    "transport_session_id", "ambient_context_surrogate",
    "ambient_context_chars", "results", "feedback_applied",
    "feedback_trace_id", "feedback_applied_at", "supersedes_bearing",
    "transcript_matched", "transcript_serialized_chars",
]
NODE_FIELDS = [
    "type", "id", "level", "scope", "included_via", "agent_surrogate",
    "agent_chars", "task_surrogate", "task_chars", "timestamp", "created_at",
    "updated_at", "decayed", "decay_reason_surrogate", "decay_reason_chars",
    "stats", "content_surrogate", "content_chars", "context_surrogate",
    "context_chars", "source_traces", "corrections", "corrections_chars",
    "provenance_shape", "relations",
]

NGRAM_LEN = 12
NGRAM_STRIDE_FIELDS = 16
NGRAM_STRIDE_RAW = 64
NGRAM_MIN_NONWS = 7
SENTINEL = "¶"  # pilcrow: asserted absent from the corpus files
K_MAX_COINCIDENCE = 3
DOC_NGRAM_LEN = 24
DOC_NGRAM_MIN_NONWS = 12


class Failures:
    def __init__(self) -> None:
        self.by_check: dict[str, list[str]] = {}

    def add(self, check: str, message: str) -> None:
        self.by_check.setdefault(check, []).append(message)

    def report(self, checks: list[str]) -> int:
        failed = 0
        for check in checks:
            messages = self.by_check.get(check, [])
            status = "FAIL" if messages else "PASS"
            print(f"[{status}] {check}" + (f" ({len(messages)} problems)" if messages else ""))
            for message in messages[:5]:
                print(f"    - {message}")
            if len(messages) > 5:
                print(f"    - ... {len(messages) - 5} more")
            failed += bool(messages)
        return failed


def sentinelize(surrogate: str) -> str:
    """Same-length copy with non-whitespace replaced by the sentinel char."""

    return "".join(ch if ch in KEEP_WS else SENTINEL for ch in surrogate)


def check_surrogate(original: str, surrogate, where: str, fails: Failures, pairs: list) -> None:
    if not isinstance(surrogate, str):
        fails.add("surrogates", f"{where}: surrogate is {type(surrogate).__name__}")
        return
    if len(surrogate) != len(original):
        fails.add("surrogates", f"{where}: length {len(surrogate)} != {len(original)}")
        return
    if not PURITY_RE.fullmatch(surrogate):
        fails.add("surrogates", f"{where}: purity violation")
        return
    for pos, (o_ch, s_ch) in enumerate(zip(original, surrogate)):
        if o_ch in KEEP_WS:
            if s_ch != o_ch:
                fails.add("surrogates", f"{where}: whitespace changed at {pos}")
                return
        elif s_ch in KEEP_WS:
            fails.add("surrogates", f"{where}: non-whitespace became whitespace at {pos}")
            return
    pairs.append((original, surrogate))


def audit_tree(original, surrogate, known_scopes, where, fails, pairs):
    """Audit a context tree; return a sentinel copy (surrogate leaves masked)."""

    if isinstance(original, dict):
        if not isinstance(surrogate, dict) or list(surrogate) != list(original):
            fails.add("surrogates", f"{where}: dict keys differ")
            return None
        return {
            k: audit_tree(original[k], surrogate[k], known_scopes, f"{where}.{k}", fails, pairs)
            for k in original
        }
    if isinstance(original, list):
        if not isinstance(surrogate, list) or len(surrogate) != len(original):
            fails.add("surrogates", f"{where}: list length differs")
            return None
        return [
            audit_tree(o, s, known_scopes, f"{where}[{i}]", fails, pairs)
            for i, (o, s) in enumerate(zip(original, surrogate))
        ]
    if isinstance(original, str):
        if keeps_real(original, known_scopes):
            if surrogate != original:
                fails.add("kept_real", f"{where}: kept-real value altered")
            return surrogate
        check_surrogate(original, surrogate, where, fails, pairs)
        return sentinelize(surrogate) if isinstance(surrogate, str) else None
    if surrogate != original:
        fails.add("kept_real", f"{where}: scalar altered")
    return surrogate


def pair_field(row, record, field, fails, pairs, where):
    """Audit one nullable private text column + its *_chars twin.

    Returns the sentinel replacement for the record's surrogate field."""

    original = row[field]
    surrogate = record[f"{field}_surrogate"]
    chars = record[f"{field}_chars"]
    if original is None:
        if surrogate is not None or chars is not None:
            fails.add("surrogates", f"{where}.{field}: null original but non-null surrogate")
        return surrogate
    if chars != len(original):
        fails.add("char_counts", f"{where}.{field}: chars {chars} != {len(original)}")
    check_surrogate(original, surrogate, f"{where}.{field}", fails, pairs)
    return sentinelize(surrogate) if isinstance(surrogate, str) else surrogate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--skip-ngram", action="store_true", help="skip checks 12-13 (slow)")
    args = parser.parse_args()
    staging, corpus_dir = args.staging, args.corpus
    fails = Failures()

    # ---- staging originals -------------------------------------------------
    alt_events = {r["id"]: r for r in read_jsonl(staging / "alt-db" / "recall_events.jsonl")}
    local_events: dict[str, dict] = {}
    for slug in LOCAL_SLUGS:
        for r in read_jsonl(staging / "local-db" / f"holdout_{slug}.recall_events.jsonl"):
            local_events[r["id"]] = r
    nodes: dict[str, dict] = {}
    for name in ("nodes.jsonl", "nodes_typed_edges.jsonl", "game_schema_nodes.jsonl",
                 "game_schema_source_nodes.jsonl"):
        for r in read_jsonl(staging / "alt-db" / name):
            nodes[r["id"]] = r
    for slug in LOCAL_SLUGS:
        for r in read_jsonl(staging / "local-db" / f"holdout_{slug}.nodes.jsonl"):
            nodes[r["id"]] = r
    typed_edges = read_jsonl(staging / "alt-db" / "connections_typed.jsonl")
    supersedes_endpoints = set()
    for e in typed_edges:
        if e["type"] == "supersedes":
            supersedes_endpoints.update((e["source_id"], e["target_id"]))
    matched_chars = {r["db_id"]: r["len_json_content"]
                     for r in read_jsonl(staging / "transcripts" / "matched.jsonl")}

    known_scopes = {"global"}
    for r in list(alt_events.values()) + list(local_events.values()):
        known_scopes.add(r["scope"])
        known_scopes.add(r["requested_scope"])
        known_scopes.update(parse_json_col(r["resolved_scopes"], []))
    for r in nodes.values():
        known_scopes.add(r["scope"])
    known_scopes = frozenset(s for s in known_scopes if s)

    # ---- corpus ------------------------------------------------------------
    splits_meta = json.loads((corpus_dir / "splits.json").read_text(encoding="utf-8"))
    rule = splits_meta["rule"]
    corpus: dict[str, dict[str, list[dict]]] = {}
    file_sha: dict[str, str] = {}
    file_lines: dict[str, list[str]] = {}
    for split in ("dev", "eval", "holdout"):
        path = corpus_dir / f"{split}.jsonl"
        raw = path.read_bytes()
        file_sha[split] = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8")
        if SENTINEL in text:
            fails.add("schema", f"{split}: sentinel char present in file")
        lines = text.splitlines()
        file_lines[split] = lines
        records = [json.loads(line) for line in lines]
        corpus[split] = {
            "node": [r for r in records if r.get("type") == "node"],
            "event": [r for r in records if r.get("type") == "event"],
        }
        other = [r for r in records if r.get("type") not in ("node", "event")]
        if other:
            fails.add("schema", f"{split}: {len(other)} records of unknown type")
        # file layout: all node records first, then all event records
        expected_layout = (["node"] * len(corpus[split]["node"])
                           + ["event"] * len(corpus[split]["event"]))
        if [r.get("type") for r in records] != expected_layout:
            fails.add("schema", f"{split}: records not in nodes-then-events order")

    pairs: list[tuple[str, str]] = []
    sentinel_records: dict[str, dict[str, list]] = {
        s: {"node": [], "event": []} for s in ("dev", "eval", "holdout")
    }

    # ---- event records -----------------------------------------------------
    def bucket(event_id: str) -> int:
        digest = hashlib.sha256(event_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % rule["mod"]

    seen_ids: Counter = Counter()
    family_seen = {"dev": 0, "eval": 0}
    for split in ("dev", "eval", "holdout"):
        node_ids_here = {n["id"] for n in corpus[split]["node"]}
        for record in corpus[split]["event"]:
            eid = record.get("id")
            where = f"{split}:{eid}"
            seen_ids[eid] += 1
            if list(record) != EVENT_FIELDS:
                fails.add("schema", f"{where}: field set/order mismatch")
                continue
            source = record["source"]
            row = (alt_events if source == "alt" else local_events).get(eid)
            if row is None:
                fails.add("splits", f"{where}: id not found in staged {source} export")
                continue
            sentinel = dict(record)
            # split rule
            if source == "local":
                expected_split = "holdout"
            else:
                b = bucket(eid)
                expected_split = ("holdout" if b < rule["holdout_bound"]
                                  else "dev" if b < rule["dev_bound"] else "eval")
            if expected_split != split:
                fails.add("splits", f"{where}: hash rule places it in {expected_split}")
            # kept-real fields
            resolved = parse_json_col(row["resolved_scopes"], [])
            results = parse_json_col(row["results"], [])
            kept = {
                "created_at": row["created_at"], "scope": row["scope"],
                "requested_scope": row["requested_scope"], "resolved_scopes": resolved,
                "depth": row["depth"], "max_results": row["max_results"],
                "results": results, "feedback_applied": row["feedback_applied"],
                "feedback_trace_id": row["feedback_trace_id"],
                "feedback_applied_at": row["feedback_applied_at"],
            }
            for key, expected in kept.items():
                if record[key] != expected:
                    fails.add("kept_real", f"{where}.{key}: differs from staging")
            transport = row["transport_session_id"]
            if transport is None or HEX32_RE.match(transport):
                if record["transport_session_id"] != transport:
                    fails.add("kept_real", f"{where}.transport_session_id: differs")
            else:
                check_surrogate(transport, record["transport_session_id"],
                                f"{where}.transport_session_id", fails, pairs)
                if isinstance(record["transport_session_id"], str):
                    sentinel["transport_session_id"] = sentinelize(record["transport_session_id"])
            # pinned predicates
            if record["class"] != classify_class(row["agent"]):
                fails.add("classes", f"{where}: class mismatch")
            if record["template_id"] != classify_template(row["query"]):
                fails.add("classes", f"{where}: template_id mismatch")
            # private fields
            for field in ("query", "agent", "task", "session_id"):
                sentinel[f"{field}_surrogate"] = pair_field(
                    row, record, field, fails, pairs, where)
            ambient = parse_json_col(row["ambient_context"], {})
            if not isinstance(ambient, dict):
                ambient = {"_raw": ambient}
            sentinel["ambient_context_surrogate"] = audit_tree(
                ambient, record["ambient_context_surrogate"],
                known_scopes, f"{where}.ambient", fails, pairs)
            if record["ambient_context_chars"] != context_chars_map(ambient):
                fails.add("char_counts", f"{where}.ambient_context_chars: mismatch")
            # flags recomputed
            bearing = bool({r.get("node_id") for r in results} & supersedes_endpoints)
            if record["supersedes_bearing"] != bearing:
                fails.add("family", f"{where}: supersedes_bearing flag wrong")
            if split in family_seen and bearing:
                family_seen[split] += 1
            expected_matched = eid in matched_chars
            if record["transcript_matched"] != expected_matched or \
               record["transcript_serialized_chars"] != matched_chars.get(eid):
                fails.add("transcripts", f"{where}: transcript match/chars mismatch")
            # containment
            refs = {r.get("node_id") for r in results if r.get("node_id")}
            if row["feedback_trace_id"]:
                refs.add(row["feedback_trace_id"])
            missing = refs - node_ids_here
            if missing:
                fails.add("containment", f"{where}: {len(missing)} referenced nodes not embedded")
            sentinel_records[split]["event"].append(sentinel)

    if seen_ids and seen_ids.most_common(1)[0][1] > 1:
        dupes = sum(1 for c in seen_ids.values() if c > 1)
        fails.add("splits", f"{dupes} event ids appear in more than one split")
    staged_ids = set(alt_events) | set(local_events)
    if set(seen_ids) != staged_ids:
        fails.add("splits", f"corpus ids != staged export ids "
                            f"(missing {len(staged_ids - set(seen_ids))}, "
                            f"extra {len(set(seen_ids) - staged_ids)})")
    for split in ("dev", "eval"):
        if family_seen[split] == 0:
            fails.add("family", f"{split}: no correction/supersedes-bearing events")

    # ---- node records ------------------------------------------------------
    for split in ("dev", "eval", "holdout"):
        for record in corpus[split]["node"]:
            nid = record.get("id")
            where = f"{split}:node:{nid}"
            if list(record) != NODE_FIELDS:
                fails.add("schema", f"{where}: field set/order mismatch")
                continue
            row = nodes.get(nid)
            if row is None:
                fails.add("kept_real", f"{where}: id not found in staging pools")
                continue
            sentinel = dict(record)
            source_traces = parse_json_col(row["source_traces"], [])
            corrections = parse_json_col(row["corrections"], [])
            provenance = parse_json_col(row["provenance"], {})
            if not isinstance(provenance, dict):
                provenance = {"_raw": provenance}
            kept = {
                "level": row["level"], "scope": row["scope"],
                "timestamp": row["timestamp"], "created_at": row["created_at"],
                "updated_at": row["updated_at"], "decayed": row["decayed"],
                "source_traces": source_traces,
            }
            for key, expected in kept.items():
                if record[key] != expected:
                    fails.add("kept_real", f"{where}.{key}: differs from staging")
            expected_stats = {
                "access_count": row["access_count"],
                "usefulness_score": row["usefulness_score"],
                "confidence": row["confidence"],
                "unique_agents": row["unique_agents"],
                "last_accessed": row["last_accessed"],
            }
            got_stats = dict(record["stats"])
            got_hint = got_stats.pop("temporal_hint", "<absent>")
            if got_stats != expected_stats:
                fails.add("kept_real", f"{where}.stats: differs from staging")
            hint = row["temporal_hint"]
            if hint is None or TEMPORAL_HINT_RE.match(hint):
                if got_hint != hint:
                    fails.add("kept_real", f"{where}.temporal_hint: differs")
            else:
                check_surrogate(hint, got_hint, f"{where}.temporal_hint", fails, pairs)
                if isinstance(got_hint, str):
                    sentinel["stats"] = {**record["stats"],
                                         "temporal_hint": sentinelize(got_hint)}
            for field in ("agent", "task"):
                sentinel[f"{field}_surrogate"] = pair_field(
                    row, record, field, fails, pairs, where)
            if row["decay_reason"] is None:
                if record["decay_reason_surrogate"] is not None or record["decay_reason_chars"] is not None:
                    fails.add("surrogates", f"{where}.decay_reason: null original, non-null surrogate")
            else:
                if record["decay_reason_chars"] != len(row["decay_reason"]):
                    fails.add("char_counts", f"{where}.decay_reason_chars: mismatch")
                check_surrogate(row["decay_reason"], record["decay_reason_surrogate"],
                                f"{where}.decay_reason", fails, pairs)
                if isinstance(record["decay_reason_surrogate"], str):
                    sentinel["decay_reason_surrogate"] = sentinelize(
                        record["decay_reason_surrogate"])
            if record["content_chars"] != len(row["content"]):
                fails.add("char_counts", f"{where}.content_chars: mismatch")
            check_surrogate(row["content"], record["content_surrogate"],
                            f"{where}.content", fails, pairs)
            if isinstance(record["content_surrogate"], str):
                sentinel["content_surrogate"] = sentinelize(record["content_surrogate"])
            context = parse_json_col(row["context"], {})
            if not isinstance(context, dict):
                context = {"_raw": context}
            sentinel["context_surrogate"] = audit_tree(
                context, record["context_surrogate"],
                known_scopes, f"{where}.context", fails, pairs)
            if record["context_chars"] != context_chars_map(context):
                fails.add("char_counts", f"{where}.context_chars: mismatch")
            # corrections: structure kept, old/new surrogated
            corr_ok = len(record["corrections"]) == len(corrections) == len(record["corrections_chars"])
            sentinel_corr = []
            if corr_ok:
                for i, (orig, got, chars) in enumerate(
                    zip(corrections, record["corrections"], record["corrections_chars"])
                ):
                    if not isinstance(orig, dict):
                        continue
                    if list(got) != list(orig):
                        fails.add("schema", f"{where}.corrections[{i}]: keys differ")
                        continue
                    entry_sentinel = {}
                    for key, value in orig.items():
                        if key in ("old", "new") and isinstance(value, str):
                            if chars.get(key) != len(value):
                                fails.add("char_counts", f"{where}.corrections[{i}].{key}: chars mismatch")
                            check_surrogate(value, got[key], f"{where}.corrections[{i}].{key}", fails, pairs)
                            entry_sentinel[key] = (sentinelize(got[key])
                                                   if isinstance(got[key], str) else got[key])
                        else:
                            entry_sentinel[key] = audit_tree(
                                value, got[key], known_scopes,
                                f"{where}.corrections[{i}].{key}", fails, pairs)
                    sentinel_corr.append(entry_sentinel)
                sentinel["corrections"] = sentinel_corr
            else:
                fails.add("schema", f"{where}.corrections: length mismatch")
            if record["provenance_shape"] != provenance_shape_of(provenance, source_traces, corrections):
                fails.add("char_counts", f"{where}.provenance_shape: mismatch")
            sentinel_records[split]["node"].append(sentinel)

    # ---- global equality classes ------------------------------------------
    orig_to_surr: dict[str, str] = {}
    surr_to_orig: dict[str, str] = {}
    for original, surrogate in pairs:
        prior = orig_to_surr.get(original)
        if prior is None:
            orig_to_surr[original] = surrogate
        elif prior != surrogate:
            fails.add("equality", f"one original got two surrogates (len {len(original)})")
        holder = surr_to_orig.get(surrogate)
        if holder is None:
            surr_to_orig[surrogate] = original
        elif holder != original:
            fails.add("equality", f"two originals share one surrogate (len {len(surrogate)})")

    # ---- splits.json bookkeeping ------------------------------------------
    for split in ("dev", "eval", "holdout"):
        node_ids = [n["id"] for n in corpus[split]["node"]]
        if len(node_ids) != len(set(node_ids)):
            fails.add("schema", f"{split}: duplicate node records in file")
        meta = splits_meta["counts"][split]
        events = corpus[split]["event"]
        actual = {
            "events": len(events),
            "nodes": len(corpus[split]["node"]),
            "by_class": dict(Counter(r["class"] for r in events)),
            "supersedes_bearing_events": sum(1 for r in events if r["supersedes_bearing"]),
            "transcript_matched_events": sum(1 for r in events if r["transcript_matched"]),
            "sha256": file_sha[split],
        }
        for key, value in actual.items():
            if meta.get(key) != value:
                fails.add("splits", f"splits.json {split}.{key} stale")
    holdout_local_scopes = {r["scope"] for r in corpus["holdout"]["event"]
                           if r["source"] == "local" and r["scope"].startswith("project:")}
    if len(holdout_local_scopes) < 2:
        fails.add("holdout", f"only {len(holdout_local_scopes)} non-AP project scopes")

    checks = ["schema", "kept_real", "char_counts", "surrogates", "equality",
              "classes", "splits", "containment", "holdout", "family",
              "transcripts"]

    # ---- n-gram leak checks ------------------------------------------------
    if not args.skip_ngram:
        checks += ["ngram_corpus", "ngram_docs"]
        leak_records = []

        # (12) corpus jsonl: sentinel-aligned adjudication ------------------
        sentinel_texts: dict[str, str] = {}
        for split in ("dev", "eval", "holdout"):
            parts = sentinel_records[split]["node"] + sentinel_records[split]["event"]
            originals = corpus[split]["node"] + corpus[split]["event"]
            lines = file_lines[split]
            if not (len(parts) == len(lines) == len(originals)):
                fails.add("schema", f"{split}: sentinel reconstruction count mismatch")
                continue
            out_lines = []
            identity_bad = 0
            for line, record, sent in zip(lines, originals, parts):
                if json.dumps(record, ensure_ascii=False) != line:
                    fails.add("schema", f"{split}: re-serialization identity broken")
                sent_line = json.dumps(sent, ensure_ascii=False)
                if len(sent_line) != len(line):
                    identity_bad += 1
                    sent_line = SENTINEL * len(line)  # conservative fallback
                out_lines.append(sent_line)
            if identity_bad:
                fails.add("schema",
                          f"{split}: {identity_bad} lines failed sentinel length identity")
            sentinel_texts[split] = "\n".join(out_lines)
            plain = "\n".join(lines)
            if len(sentinel_texts[split]) != len(plain):
                fails.add("schema", f"{split}: sentinel text length mismatch")

        shingles: set[str] = set()

        def sample(text: str, stride: int) -> None:
            for start in range(0, max(1, len(text) - NGRAM_LEN + 1), stride):
                gram = text[start:start + NGRAM_LEN]
                if len(gram) == NGRAM_LEN and sum(1 for c in gram if c not in KEEP_WS) >= NGRAM_MIN_NONWS:
                    shingles.add(gram)

        for original in orig_to_surr:
            sample(original, NGRAM_STRIDE_FIELDS)
        raw_sources = [staging / "alt-db" / n for n in (
            "recall_events.jsonl", "nodes.jsonl", "nodes_typed_edges.jsonl",
            "game_schema_nodes.jsonl", "game_schema_source_nodes.jsonl",
            "outcome_nodes.jsonl", "connections_typed.jsonl")]
        raw_sources += [staging / "local-db" / f"holdout_{slug}.{kind}.jsonl"
                        for slug in LOCAL_SLUGS for kind in ("recall_events", "nodes")]
        raw_sources.append(staging / "transcripts" / "lm_events.jsonl")
        for path in raw_sources:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    sample(line.rstrip("\n"), NGRAM_STRIDE_RAW)

        k_histogram: Counter = Counter()
        hits = 0
        for split in ("dev", "eval", "holdout"):
            text = "\n".join(file_lines[split])
            sent = sentinel_texts.get(split, "")
            aligned = len(sent) == len(text)
            for i in range(len(text) - NGRAM_LEN + 1):
                gram = text[i:i + NGRAM_LEN]
                if gram not in shingles:
                    continue
                hits += 1
                k = sent[i:i + NGRAM_LEN].count(SENTINEL) if aligned else NGRAM_LEN
                k_histogram[k] += 1
                if k > K_MAX_COINCIDENCE:
                    leak_records.append({"check": "corpus", "file": f"{split}.jsonl",
                                         "offset": i, "k": k, "gram": gram})
        print(f"ngram_corpus: {len(shingles)} sampled shingles, {hits} window hits, "
              f"k-histogram {dict(sorted(k_histogram.items()))}, "
              f"{sum(1 for e in leak_records if e['check'] == 'corpus')} leaks (k>{K_MAX_COINCIDENCE})")

        # (13) authored files: exhaustive 24-grams vs full staging text -----
        doc_paths = [corpus_dir / "splits.json", corpus_dir / "POLICY.md",
                     HERE.parent / "02-transform.md"] + sorted(HERE.glob("*.py"))
        doc_grams: dict[str, list[str]] = {}
        for path in doc_paths:
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            for i in range(len(text) - DOC_NGRAM_LEN + 1):
                gram = text[i:i + DOC_NGRAM_LEN]
                if sum(1 for c in gram if c not in KEEP_WS) >= DOC_NGRAM_MIN_NONWS:
                    doc_grams.setdefault(gram, []).append(path.name)
        doc_hits: set[str] = set()
        staging_stream = raw_sources + [staging / "deid_map" / "surrogates.jsonl"]
        for path in staging_stream:
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            for i in range(len(text) - DOC_NGRAM_LEN + 1):
                gram = text[i:i + DOC_NGRAM_LEN]
                if gram in doc_grams:
                    doc_hits.add(gram)

        # A hit is benign iff every alphanumeric token in the gram is a
        # fragment of kept-real-by-design vocabulary: corpus field/key names,
        # kept-real (non-sentinel) string leaves (ids, scopes, timestamps,
        # template ids, enums), or a digit/timestamp fragment.  Punctuation
        # and JSON structure carry no content.
        vocab: set[str] = set(EVENT_FIELDS) | set(NODE_FIELDS)

        def collect_vocab(value) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    vocab.add(key)
                    collect_vocab(item)
            elif isinstance(value, list):
                for item in value:
                    collect_vocab(item)
            elif isinstance(value, str) and SENTINEL not in value:
                vocab.add(value)

        for split in ("dev", "eval", "holdout"):
            for sent in sentinel_records[split]["node"] + sentinel_records[split]["event"]:
                collect_vocab(sent)
        token_re = re.compile(r"[A-Za-z0-9_]+")
        ts_fragment_re = re.compile(r"[0-9][0-9TZ]*")

        def benign_doc_gram(gram: str) -> bool:
            for token in token_re.findall(gram):
                if ts_fragment_re.fullmatch(token):
                    continue
                if any(token in item for item in vocab):
                    continue
                return False
            return True

        doc_leaks = {gram for gram in doc_hits if not benign_doc_gram(gram)}
        for gram in doc_leaks:
            leak_records.append({"check": "docs", "files": sorted(set(doc_grams[gram])),
                                 "gram": gram})
        print(f"ngram_docs: {len(doc_grams)} doc 24-grams checked exhaustively, "
              f"{len(doc_hits)} found in staging text, "
              f"{len(doc_hits) - len(doc_leaks)} explained by kept-real vocabulary, "
              f"{len(doc_leaks)} leaks")

        diag = staging / "deid_map" / "ngram_leaks.jsonl"
        if not leak_records:
            diag.unlink(missing_ok=True)  # clear stale diagnostics
        else:
            diag.parent.mkdir(parents=True, exist_ok=True)
            with diag.open("w", encoding="utf-8") as fh:
                for entry in leak_records:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            corpus_leaks = [e for e in leak_records if e["check"] == "corpus"]
            if corpus_leaks:
                fails.add("ngram_corpus",
                          f"{len(corpus_leaks)} windows with k>{K_MAX_COINCIDENCE}; "
                          f"grams in {diag}")
            if doc_leaks:
                fails.add("ngram_docs",
                          f"{len(doc_leaks)} authored 24-grams quote staged private "
                          f"text; grams in {diag}")

    failed = fails.report(checks)
    print(f"pairs audited: {len(pairs)}; unique originals: {len(orig_to_surr)}")
    print("VERIFY:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
