#!/usr/bin/env python3
"""Recalibrate the grounding threshold on the closed-recall corpora, criterion first.

``living_memory.grounding.DEFAULT_MIN_CONTAINMENT`` was fixed at 0.25 on the
June snapshot (``artifacts/grounding/calibration.md``) as the point where the
live per-event IDF view agrees with the whole-corpus view; that was a
consistency argument, not a signal/noise one. Since then traces grew threefold
and the remember protocol forbids copies, so on the September corpora only a
few percent of result/trace pairs clear 0.25 and most closures reinforce
nothing (``artifacts/grounding/usage-signal-baseline.md``). This script picks
the threshold again, on those corpora, under the final tokenizer, with the
criterion below — stated here, before the sweep is run, so the number cannot
be chosen to flatter a result.

Everything is measured through ``scripts/usage_signal_replay.py`` (imported as
a module; nothing about grounding or relatedness is re-derived here). The
external reference is the production multilingual encoder: a pair is
*related* when the cosine between the delivered node and the closing trace
is at or above the cut, *unrelated* when at or below the lower cut, and the
lexical tokenizer under test contributes nothing to that verdict.

Selection criterion
-------------------

Populations: pooled, and each host on its own (``sfx``, ``alt``) — the live
gate runs per host, against that host's own learned weights, so a threshold
has to be clean on each host, not on their average.

Relatedness cuts, both applied: related >= 0.5 / unrelated <= 0.3 and
related >= 0.6 / unrelated <= 0.4.

For a threshold ``t``, a population and a cut, over result/trace pairs:

* ``noise(t)``  — grounded share among *unrelated* pairs (the lexical
  false-positive rate the gate must hold down);
* ``signal(t)`` — grounded share among *related* pairs;
* ``ratio(t)``  — ``signal / noise`` (infinite when noise is 0 and signal
  positive; undefined, and failing, when both are 0).

``t`` **clears** the criterion when, for every population and both cuts,
``noise(t) <= NOISE_CEILING`` (5%) and ``ratio(t) >= SNR_FLOOR`` (5). A class
with fewer than ``MIN_CLASS_PAIRS`` (50) related or unrelated pairs is reported
but does not gate.

**Adopt the lowest ``t`` in the sweep (0.05..0.30, step 0.01) such that ``t``
and every higher threshold in the sweep clear the criterion** — the start of
the terminal passing run — so a one-step dip caused by a pair or two cannot
select a threshold its neighbour contradicts. If no threshold clears, the
shipped 0.25 stays and the artifact says so.

Closure-level figures (share of related vs unrelated *closures* with at least
one grounded result) are reported beside every row and do not gate: the
unrelated closure classes are small (86 pooled at 0.5/0.3) and a closure
aggregates about eight results, so its noise floor is structurally higher
than the pair-level one the credit rule actually decides on.

Pre-registration note. Before this script was written the coarse figures at
0.10/0.15/0.20/0.25 under the new tokenizer, at the 0.5/0.3 cut only, were
already published in ``artifacts/grounding/cyrillic-tokenizer.md``. The
criterion above was fixed with those in view and was not adjusted after the
0.01-step sweep or the 0.6/0.4 cut were run.

Holdout replay
--------------

With the adopted value known, ``scripts/credit_rule_ab.py`` is run unchanged
on the sfx snapshot at the old and the adopted threshold (the threshold enters
through ``LM_GROUNDING_MIN_CONTAINMENT``, which ``living_memory.grounding``
reads at import), at each ``--holdout-cutoff``. The grounded arm's holdout
hit@5/MRR are tabulated under the grounded label and under the two labels that
do not derive from grounding (``reconsumed``, ``usefulness``). Under the
grounded label the *label itself* moves with the threshold, so only the
within-run paired delta against the incumbent is meaningful there; the
control labels are threshold-independent, so the grounded arm's figures at
the two thresholds are directly comparable and the incumbent's must be
identical (that identity is checked and reported).

Usage::

    python3 scripts/grounding_recalibration.py \
        --holdout-db ~/.cache/living-memory-harness/usage-signal/sfx-2026-09-07.sqlite3

    python3 scripts/grounding_recalibration.py --skip-holdout   # sweep only
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))

import usage_signal_replay as usr  # noqa: E402
from living_memory import replay as lm_replay  # noqa: E402
from living_memory.embeddings import cyrillic_stem_enabled  # noqa: E402
from living_memory.grounding import (  # noqa: E402
    CALIBRATED_MIN_CONTAINMENT,
    DEFAULT_MIN_CONTAINMENT,
    MIN_CONTAINMENT_ENV_VAR,
)

# --- criterion constants (see the module docstring) ------------------------
PREVIOUS_MIN_CONTAINMENT = 0.25
CANDIDATE_MIN_CONTAINMENT = 0.15
NOISE_CEILING = 0.05
SNR_FLOOR = 5.0
MIN_CLASS_PAIRS = 50
CUTS: tuple[tuple[float, float], ...] = ((0.5, 0.3), (0.6, 0.4))
SWEEP_START, SWEEP_STOP, SWEEP_STEP = 0.05, 0.30, 0.01

# --- corpora -----------------------------------------------------------------
HOSTS = ("sfx", "alt")
CORPUS_DIR = Path.home() / ".cache" / "living-memory-harness" / "usage-signal"
SNAPSHOT_DATE = "2026-09-07"
CORPUS_SINCE = "2026-09-04T00:00:00Z"
POPULATIONS = ("pooled", *HOSTS)

# --- holdout -----------------------------------------------------------------
DEFAULT_HOLDOUT_CUTOFFS = ("2026-09-04T00:00:00Z", "2026-08-24T00:00:00Z")
DEFAULT_HOLDOUT_DIR = ROOT / "artifacts" / "grounding" / "recalibration-2026-09-inputs"
CONTROL_LABELS = ("reconsumed", "usefulness")
PRIMARY_LABEL = "grounded"
GROUNDED_ARM = "grounded"
INCUMBENT_ARM = "proportional"

DEFAULT_OUT_JSON = ROOT / "artifacts" / "grounding" / "recalibration-2026-09.json"
DEFAULT_OUT_MD = ROOT / "artifacts" / "grounding" / "recalibration-2026-09.md"


def sweep_thresholds() -> list[float]:
    steps = int(round((SWEEP_STOP - SWEEP_START) / SWEEP_STEP))
    return [round(SWEEP_START + index * SWEEP_STEP, 2) for index in range(steps + 1)]


def ensure_corpus(host: str, corpus_dir: Path) -> Path:
    """The host's closure corpus, regenerated from its snapshot only if missing."""

    path = corpus_dir / f"{host}-closures.jsonl"
    if path.exists():
        return path
    snapshot = corpus_dir / f"{host}-{SNAPSHOT_DATE}.sqlite3"
    if not snapshot.exists():
        raise SystemExit(f"missing corpus {path} and snapshot {snapshot}; nothing to regenerate from")
    print(f"regenerating {path} from {snapshot}", file=sys.stderr)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "usage_signal_corpus.py"),
            "--db",
            str(snapshot),
            "--since",
            CORPUS_SINCE,
            "--host",
            host,
            "--out",
            str(path),
        ],
        check=True,
    )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


# ---------------------------------------------------------------------------
# Sweep and criterion
# ---------------------------------------------------------------------------


def _rows_by_population(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = {"pooled": report["pooled"]}
    rows.update(report["per_host"])
    return rows


def _check(row: dict[str, Any]) -> dict[str, Any]:
    """One population/cut/threshold against the criterion, from a replay row."""

    related = row["pairs_by_relatedness"].get("related", {"n": 0, "hit": 0, "share": 0.0})
    unrelated = row["pairs_by_relatedness"].get("unrelated", {"n": 0, "hit": 0, "share": 0.0})
    ratio = row["pair_signal_to_noise"]["ratio"]
    gated = related["n"] >= MIN_CLASS_PAIRS and unrelated["n"] >= MIN_CLASS_PAIRS
    noise_ok = unrelated["share"] <= NOISE_CEILING
    ratio_ok = ratio is not None and ratio >= SNR_FLOOR
    return {
        "related": related,
        "unrelated": unrelated,
        "ratio": ratio,
        "gated": gated,
        "noise_ok": noise_ok,
        "ratio_ok": ratio_ok,
        "pass": (noise_ok and ratio_ok) if gated else True,
        "closures": {
            "all": row["closures"],
            "related": row["closures_by_relatedness"].get("related"),
            "unrelated": row["closures_by_relatedness"].get("unrelated"),
            "ratio": row["closure_signal_to_noise"]["ratio"],
        },
    }


def run_sweep(records: Sequence[dict[str, Any]], thresholds: Sequence[float]) -> dict[str, Any]:
    """Replay both cuts, evaluate the criterion per threshold, pick the value."""

    cuts: dict[str, dict[str, Any]] = {}
    for related, unrelated in CUTS:
        key = f"{related}/{unrelated}"
        report = usr.replay(
            records,
            thresholds=thresholds,
            related=related,
            unrelated=unrelated,
            lookup_window=usr.DEFAULT_LOOKUP_WINDOW_SECONDS,
            same_transport=True,
        )
        rows = _rows_by_population(report)
        cuts[key] = {
            "related_cosine": related,
            "unrelated_cosine": unrelated,
            "rows": rows,
            "class_sizes": {
                population: _class_sizes(rows[population][0]) for population in POPULATIONS
            },
            "pair_containment_percentiles": report["pair_containment_percentiles"],
        }

    per_threshold: list[dict[str, Any]] = []
    for index, threshold in enumerate(thresholds):
        checks: dict[str, dict[str, Any]] = {}
        for key, block in cuts.items():
            checks[key] = {
                population: _check(block["rows"][population][index])
                for population in POPULATIONS
            }
        failures = [
            f"{population}@{key}"
            for key, block in checks.items()
            for population, check in block.items()
            if not check["pass"]
        ]
        per_threshold.append(
            {
                "threshold": threshold,
                "checks": checks,
                "clears": not failures,
                "failures": failures,
            }
        )

    # Terminal passing run: lowest t such that t and every higher t clear.
    adopted: float | None = None
    for entry in reversed(per_threshold):
        if entry["clears"]:
            adopted = entry["threshold"]
        else:
            break
    return {
        "thresholds": list(thresholds),
        "cuts": cuts,
        "per_threshold": per_threshold,
        "adopted": adopted,
        "adopted_is_terminal_run_start": adopted is not None,
    }


def _class_sizes(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "pairs": {
            band: row["pairs_by_relatedness"].get(band, {}).get("n", 0)
            for band in ("related", "middle", "unrelated")
        },
        "closures": {
            band: row["closures_by_relatedness"].get(band, {}).get("n", 0)
            for band in ("related", "middle", "unrelated")
        },
        "pairs_total": row["pairs"]["n"],
        "closures_total": row["closures"]["n"],
    }


def _entry(sweep: dict[str, Any], threshold: float) -> dict[str, Any] | None:
    for entry in sweep["per_threshold"]:
        if abs(entry["threshold"] - threshold) < 1e-9:
            return entry
    return None


def explain_failure(sweep: dict[str, Any], threshold: float) -> list[str]:
    """Human-readable reasons a threshold does not clear, with denominators."""

    entry = _entry(sweep, threshold)
    if entry is None:
        return [f"{threshold} is not in the sweep"]
    reasons: list[str] = []
    for key, block in entry["checks"].items():
        for population, check in block.items():
            if check["pass"]:
                continue
            unrelated = check["unrelated"]
            related = check["related"]
            parts = []
            if not check["noise_ok"]:
                parts.append(
                    f"unrelated pairs grounded {unrelated['hit']}/{unrelated['n']} "
                    f"({unrelated['share']:.1%}) > {NOISE_CEILING:.0%}"
                )
            if not check["ratio_ok"]:
                parts.append(
                    f"related/unrelated ratio {check['ratio']} < {SNR_FLOOR:g} "
                    f"(related {related['hit']}/{related['n']} = {related['share']:.1%})"
                )
            reasons.append(f"{population} at cut {key}: " + "; ".join(parts))
    if not reasons and not entry["clears"]:
        reasons.append("fails only through the terminal-run rule (a higher threshold fails)")
    if entry["clears"] and sweep["adopted"] is not None and threshold < sweep["adopted"]:
        reasons.append(
            "clears on its own but a higher threshold in the sweep does not, so it is not "
            "the start of the terminal passing run"
        )
    return reasons


# ---------------------------------------------------------------------------
# Coverage and anchors
# ---------------------------------------------------------------------------


def _row_at(sweep: dict[str, Any], cut_key: str, population: str, threshold: float) -> dict[str, Any]:
    rows = sweep["cuts"][cut_key]["rows"][population]
    for row in rows:
        if abs(row["threshold"] - threshold) < 1e-9:
            return row
    raise KeyError(threshold)


def coverage_change(sweep: dict[str, Any], old: float, new: float) -> dict[str, Any]:
    """Grounded pairs/closures at the old vs the adopted threshold (cut-independent)."""

    cut_key = next(iter(sweep["cuts"]))
    out: dict[str, Any] = {"old": old, "new": new, "populations": {}}
    for population in POPULATIONS:
        before = _row_at(sweep, cut_key, population, old)
        after = _row_at(sweep, cut_key, population, new)
        out["populations"][population] = {
            "pairs": {"old": before["pairs"], "new": after["pairs"]},
            "pairs_lat_lat": {
                "old": before["pairs_by_script"].get("lat/lat"),
                "new": after["pairs_by_script"].get("lat/lat"),
            },
            "pairs_cyr_any": {
                "old": before["pairs_by_script"].get("cyr-any"),
                "new": after["pairs_by_script"].get("cyr-any"),
            },
            "closures": {"old": before["closures"], "new": after["closures"]},
            "closures_grounded_or_looked_up": {
                "old": before["lookups"]["closures_grounded_or_looked_up"],
                "new": after["lookups"]["closures_grounded_or_looked_up"],
            },
            "lookup_pairs_also_grounded": {
                "old": before["lookups"]["overlap_pairs_grounded_and_looked_up"],
                "new": after["lookups"]["overlap_pairs_grounded_and_looked_up"],
            },
        }
    return out


def anchor_change(sweep: dict[str, Any], old: float, new: float) -> dict[str, Any]:
    """Anchor edges the live path would write: one per grounded result of a closure.

    ``feedback._reinforce_query_anchors`` writes the consumed event's query as
    an anchor with an edge to each grounded result, so grounded results per
    closure is the edge count and closures with any grounded result is the
    anchor count (before dedup against an existing anchor for the same query).
    """

    cut_key = next(iter(sweep["cuts"]))
    out: dict[str, Any] = {"old": old, "new": new, "populations": {}}
    for population in POPULATIONS:
        block: dict[str, Any] = {}
        for label, threshold in (("old", old), ("new", new)):
            row = _row_at(sweep, cut_key, population, threshold)
            closures = row["closures"]["n"]
            grounding_closures = row["closures"]["hit"]
            edges = row["pairs"]["hit"]
            block[label] = {
                "closures": closures,
                "closures_writing_an_anchor": grounding_closures,
                "anchor_edges": edges,
                "edges_per_closure": round(edges / closures, 4) if closures else 0.0,
                "edges_per_anchor_writing_closure": (
                    round(edges / grounding_closures, 4) if grounding_closures else 0.0
                ),
            }
        block["delta"] = {
            "closures_writing_an_anchor": block["new"]["closures_writing_an_anchor"]
            - block["old"]["closures_writing_an_anchor"],
            "anchor_edges": block["new"]["anchor_edges"] - block["old"]["anchor_edges"],
            "edges_per_closure": round(
                block["new"]["edges_per_closure"] - block["old"]["edges_per_closure"], 4
            ),
        }
        out["populations"][population] = block
    return out


# ---------------------------------------------------------------------------
# Holdout replay (credit_rule_ab.py, unchanged, threshold via the environment)
# ---------------------------------------------------------------------------


def _schema_aware_snapshot_evidence(
    connection: sqlite3.Connection,
) -> Callable[[str], tuple[bool, bool]]:
    """``replay.snapshot_evidence`` for snapshots without ``nodes.embedding``.

    ``replay.snapshot_evidence`` predates the chunked embedding store and asks
    ``nodes.embedding``, which the 2026-09 snapshots no longer have, so the
    A/B crashes on them. When the column exists the original is used
    untouched; otherwise the vector gate mirrors
    ``MemoryStore._has_vector_evidence`` (a live chunk embedding for a live
    node of the scope) and the graph gate is the original query. This is the
    evidence *gate* only — a per-scope boolean the weight floors consult — and
    not the credit arithmetic. The lookup-credit branch carries the same fix
    inside ``replay.snapshot_evidence`` itself; once it lands this shim is a
    pass-through on every snapshot.
    """

    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(nodes)")}
    if "embedding" in columns:
        return lm_replay.snapshot_evidence(connection)
    from living_memory.storage import CHUNK_EMBEDDING_TABLE

    chunk_table = (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (CHUNK_EMBEDDING_TABLE,),
        ).fetchone()
        is not None
    )
    cache: dict[str, tuple[bool, bool]] = {}

    def evidence(scope: str) -> tuple[bool, bool]:
        cached = cache.get(scope)
        if cached is not None:
            return cached
        vector = False
        if chunk_table:
            vector = (
                connection.execute(
                    f"""
                    SELECT 1
                    FROM {CHUNK_EMBEDDING_TABLE} c
                    JOIN nodes n ON n.id = c.node_id
                    WHERE n.scope = ? AND n.decayed = 0
                    LIMIT 1
                    """,
                    (scope,),
                ).fetchone()
                is not None
            )
        graph_row = connection.execute(
            """
            SELECT 1
            FROM connections c
            JOIN nodes source ON source.id = c.source_id
            JOIN nodes target ON target.id = c.target_id
            WHERE source.decayed = 0 AND target.decayed = 0
              AND (source.scope = ? OR target.scope = ?)
            LIMIT 1
            """,
            (scope, scope),
        ).fetchone()
        result = (vector, graph_row is not None)
        cache[scope] = result
        return result

    return evidence


def _load_credit_rule_ab() -> Any:
    spec = importlib.util.spec_from_file_location("credit_rule_ab", SCRIPTS / "credit_rule_ab.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["credit_rule_ab"] = module
    spec.loader.exec_module(module)
    return module


def credit_ab_worker(argv: Sequence[str]) -> int:
    """Run ``credit_rule_ab.main`` in this process with the evidence shim applied.

    The threshold is whatever ``LM_GROUNDING_MIN_CONTAINMENT`` said when this
    interpreter imported ``living_memory``; the parent sets it on the
    subprocess environment, so ``credit_rule_ab.py`` itself is untouched.
    """

    credit_rule_ab = _load_credit_rule_ab()
    credit_rule_ab.snapshot_evidence = _schema_aware_snapshot_evidence
    return int(credit_rule_ab.main(list(argv)))


def holdout_report_path(holdout_dir: Path, threshold: float, cutoff: str) -> Path:
    return holdout_dir / f"credit-ab-min{threshold:.2f}-cut{cutoff[:10]}.json"


def run_holdout(
    db: Path, cutoff: str, threshold: float, holdout_dir: Path, *, force: bool = False
) -> dict[str, Any]:
    """Run (or reuse) one credit_rule_ab replay at ``threshold`` and load it."""

    path = holdout_report_path(holdout_dir, threshold, cutoff)
    if force or not path.exists():
        holdout_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env[MIN_CONTAINMENT_ENV_VAR] = f"{threshold:g}"
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (env.get("PYTHONPATH", ""), str(ROOT / ".cache" / "python-deps")) if part
        )
        print(
            f"credit_rule_ab: db={db} cutoff={cutoff} {MIN_CONTAINMENT_ENV_VAR}={threshold:g} -> {path}",
            file=sys.stderr,
        )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--credit-ab-worker",
                "--db",
                str(db),
                "--cutoff",
                cutoff,
                "--report",
                str(path),
                "--markdown",
                str(path.with_suffix(".md")),
            ],
            check=True,
            env=env,
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    actual = report["grounding"]["min_containment"]
    if abs(actual - threshold) > 1e-9:
        raise SystemExit(
            f"{path} was graded at min_containment {actual}, expected {threshold}; "
            f"the {MIN_CONTAINMENT_ENV_VAR} override did not reach the replay"
        )
    return report


def summarize_holdout(
    reports: dict[str, dict[str, dict[str, Any]]], old: float, new: float
) -> dict[str, Any]:
    """Old-vs-new tables per cutoff, and whether the controls move against ``new``."""

    summary: dict[str, Any] = {"old": old, "new": new, "cutoffs": {}}
    for cutoff, by_threshold in reports.items():
        block: dict[str, Any] = {"labels": {}, "grounding": {}, "weight_sanity": {}, "verdict": {}}
        for label, report in by_threshold.items():
            block["grounding"][label] = report["grounding"]
            block["weight_sanity"][label] = report["weight_sanity"][GROUNDED_ARM]
            block["verdict"][label] = {
                "winner": report["verdict"]["winner"],
                "grounded_arm": {
                    key: report["verdict"]["per_arm"][GROUNDED_ARM][key]
                    for key in ("pass", "significant_regressions", "primary_gains", "weights_degenerate")
                },
            }
            block["usable_events"] = report["corpus"]["usable_events"]
        for label_protocol in (PRIMARY_LABEL, *CONTROL_LABELS):
            rows: dict[str, Any] = {}
            for label, report in by_threshold.items():
                arms = report["results"][label_protocol]["arms"]
                rows[label] = {
                    arm: {
                        "hit@5": arms[arm]["holdout"]["hit@5"],
                        "mrr": arms[arm]["holdout"]["mrr"],
                        "events_with_useful": arms[arm]["holdout"]["events_with_useful"],
                        "events": arms[arm]["holdout"]["events"],
                        "weight_updates": arms[arm]["weight_updates"],
                    }
                    for arm in (INCUMBENT_ARM, GROUNDED_ARM)
                }
                paired = arms[GROUNDED_ARM]["paired_vs_incumbent"]
                checks = report["verdict"]["per_arm"][GROUNDED_ARM]["checks"][label_protocol]
                rows[label]["grounded_vs_incumbent"] = {
                    metric: {
                        "delta": checks[metric]["delta"],
                        "ci95": checks[metric]["ci95"],
                        "distinguishable": checks[metric]["distinguishable"],
                        "regression": checks[metric]["regression"],
                        "gain": checks[metric]["gain"],
                        "n": paired[metric]["n"],
                    }
                    for metric in ("hit@5", "mrr")
                }
            old_row, new_row = rows.get("old"), rows.get("new")
            if old_row and new_row:
                rows["new_minus_old"] = {
                    "grounded_arm": {
                        metric: round(
                            new_row[GROUNDED_ARM][metric] - old_row[GROUNDED_ARM][metric], 6
                        )
                        for metric in ("hit@5", "mrr")
                    },
                    "incumbent_arm": {
                        metric: round(
                            new_row[INCUMBENT_ARM][metric] - old_row[INCUMBENT_ARM][metric], 6
                        )
                        for metric in ("hit@5", "mrr")
                    },
                }
            block["labels"][label_protocol] = rows
        # Controls are threshold-independent labels: the incumbent must not move,
        # and the grounded arm's move is the threshold's effect.
        controls: dict[str, Any] = {}
        for control in CONTROL_LABELS:
            rows = block["labels"][control]
            if "new_minus_old" not in rows:
                continue
            incumbent_moved = any(
                abs(value) > 1e-9 for value in rows["new_minus_old"]["incumbent_arm"].values()
            )
            grounded_delta = rows["new_minus_old"]["grounded_arm"]
            events_with_useful = rows["new"][GROUNDED_ARM]["events_with_useful"]
            controls[control] = {
                "incumbent_identical_across_thresholds": not incumbent_moved,
                "events_with_useful": events_with_useful,
                "grounded_arm_new_minus_old": grounded_delta,
                # hit@5 is a share of events, so its delta has an exact event count.
                "grounded_arm_new_minus_old_hit5_events": round(
                    grounded_delta["hit@5"] * events_with_useful
                ),
                "moves_against_new": any(value < 0.0 for value in grounded_delta.values()),
                "old_arm_vs_incumbent": rows["old"]["grounded_vs_incumbent"],
                "new_arm_vs_incumbent": rows["new"]["grounded_vs_incumbent"],
            }
        block["controls"] = controls
        block["controls_move_against_new"] = any(
            item["moves_against_new"] for item in controls.values()
        )
        block["controls_significant_regression_at_new"] = any(
            item["new_arm_vs_incumbent"][metric]["regression"]
            for item in controls.values()
            for metric in ("hit@5", "mrr")
        )
        summary["cutoffs"][cutoff] = block
    return summary


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cell(tally: dict[str, Any] | None) -> str:
    if not tally:
        return "–"
    return f"{tally['hit']}/{tally['n']} ({tally['share']:.1%})"


def _regressions(checks: dict[str, Any]) -> str:
    return ", ".join(metric for metric in ("hit@5", "mrr") if checks[metric]["regression"])


def _ratio_cell(ratio: float | None) -> str:
    if ratio is None:
        return "n/a"
    if ratio == float("inf"):
        return "inf"
    return f"{ratio:.2f}"


def render_markdown(result: dict[str, Any]) -> str:
    sweep = result["sweep"]
    adopted = result["adopted"]
    old = PREVIOUS_MIN_CONTAINMENT
    lines: list[str] = []
    lines.append(
        f"# Grounding threshold recalibration (snapshots {SNAPSHOT_DATE}, closures since {CORPUS_SINCE[:10]})"
    )
    lines.append("")
    lines.append(
        f"- Checkout `{result['checkout']['commit']}`"
        f"{' (dirty)' if result['checkout']['dirty'] else ''}; tokenizer Cyrillic stemming "
        f"{'on' if result['tokenizer']['cyrillic_stem_enabled'] else 'off'}; "
        f"process default `DEFAULT_MIN_CONTAINMENT` = {result['tokenizer']['default_min_containment']}, "
        f"shipped constant `CALIBRATED_MIN_CONTAINMENT` = {result['tokenizer']['calibrated_min_containment']}."
    )
    for corpus in result["corpora"]:
        lines.append(
            f"- Corpus `{corpus['host']}`: {corpus['records']} closures / {corpus['pairs']} pairs, "
            f"`{corpus['path']}` (sha256 {corpus['sha256'][:12]}…)."
        )
    lines.append(
        f"- Sweep {SWEEP_START:.2f}..{SWEEP_STOP:.2f} step {SWEEP_STEP:.2f} "
        f"({len(sweep['thresholds'])} thresholds); populations {', '.join(POPULATIONS)}; "
        f"cuts {', '.join(sweep['cuts'])}."
    )
    lines.append("")
    lines.append("## Criterion (fixed before the sweep)")
    lines.append("")
    lines.append(
        f"Over result/trace pairs, for every population ({', '.join(POPULATIONS)}) and both "
        f"relatedness cuts: grounded share among unrelated pairs ≤ {NOISE_CEILING:.0%} and "
        f"related/unrelated grounded-share ratio ≥ {SNR_FLOOR:g}; classes under {MIN_CLASS_PAIRS} "
        "pairs are reported but do not gate. Adopt the lowest threshold such that it and every "
        "higher threshold in the sweep clear (start of the terminal passing run). Closure-level "
        "figures are reported beside each row and do not gate. Full statement and the "
        "pre-registration note: `scripts/grounding_recalibration.py` docstring."
    )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if adopted is None:
        lines.append(
            f"- **No threshold in the sweep clears the criterion; `DEFAULT_MIN_CONTAINMENT` stays at {old}.**"
        )
    else:
        lines.append(f"- **Adopted: {adopted}** (previous {old}).")
        pooled = result["coverage_change"]["populations"]["pooled"]
        edges = result["anchors"]["populations"]["pooled"]
        lines.append(
            f"- Pooled coverage {old} → {adopted}: pairs grounded {_cell(pooled['pairs']['old'])} → "
            f"{_cell(pooled['pairs']['new'])}; closures reinforcing at least one node "
            f"{_cell(pooled['closures']['old'])} → {_cell(pooled['closures']['new'])}; anchor edges "
            f"per closure {edges['old']['edges_per_closure']:.3f} → {edges['new']['edges_per_closure']:.3f} "
            "(per-host rows below)."
        )
    lines.append(
        f"- Thresholds that clear: "
        f"{', '.join(f'{e['threshold']:.2f}' for e in sweep['per_threshold'] if e['clears']) or 'none'}."
    )
    lines.append(f"- Why {CANDIDATE_MIN_CONTAINMENT} {'clears' if _entry(sweep, CANDIDATE_MIN_CONTAINMENT) and _entry(sweep, CANDIDATE_MIN_CONTAINMENT)['clears'] else 'fails'}:")
    for reason in result["candidate_0_15"]["reasons"] or ["clears every check"]:
        lines.append(f"  - {reason}")
    if adopted is not None and adopted != old:
        lines.append(f"- First failing check of each threshold below {adopted}:")
        for entry in sweep["per_threshold"]:
            if entry["threshold"] >= adopted:
                break
            reasons = explain_failure(sweep, entry["threshold"])
            lines.append(f"  - {entry['threshold']:.2f}: {reasons[0] if reasons else 'clears'}")
    lines.append("")
    lines.append("## Class sizes")
    lines.append("")
    lines.append("| cut | population | pairs related | pairs middle | pairs unrelated | closures related | closures middle | closures unrelated |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for key, block in sweep["cuts"].items():
        for population in POPULATIONS:
            sizes = block["class_sizes"][population]
            lines.append(
                f"| {key} | {population} | {sizes['pairs']['related']} | {sizes['pairs']['middle']} "
                f"| {sizes['pairs']['unrelated']} | {sizes['closures']['related']} "
                f"| {sizes['closures']['middle']} | {sizes['closures']['unrelated']} |"
            )
    lines.append("")
    for key, block in sweep["cuts"].items():
        lines.append(f"## Sweep at cut {key} — pairs (the gating level)")
        lines.append("")
        header = "| thr |"
        sep = "|---|"
        for population in POPULATIONS:
            header += f" {population} unrelated | {population} related | {population} ratio |"
            sep += "---|---|---|"
        header += " clears |"
        sep += "---|"
        lines.append(header)
        lines.append(sep)
        for entry in sweep["per_threshold"]:
            row = f"| {entry['threshold']:.2f} |"
            for population in POPULATIONS:
                check = entry["checks"][key][population]
                mark = "" if check["pass"] else " ✗"
                row += (
                    f" {_cell(check['unrelated'])}{'' if check['noise_ok'] else ' ✗'} "
                    f"| {_cell(check['related'])} | {_ratio_cell(check['ratio'])}{'' if check['ratio_ok'] else ' ✗'} |"
                )
                _ = mark
            row += f" {'yes' if entry['clears'] else 'no'} |"
            lines.append(row)
        lines.append("")
        lines.append(f"## Sweep at cut {key} — closures (reported, not gating)")
        lines.append("")
        header = "| thr |"
        sep = "|---|"
        for population in POPULATIONS:
            header += f" {population} grounded | {population} unrelated | {population} related | {population} ratio |"
            sep += "---|---|---|---|"
        lines.append(header)
        lines.append(sep)
        for entry in sweep["per_threshold"]:
            row = f"| {entry['threshold']:.2f} |"
            for population in POPULATIONS:
                closures = entry["checks"][key][population]["closures"]
                row += (
                    f" {_cell(closures['all'])} | {_cell(closures['unrelated'])} "
                    f"| {_cell(closures['related'])} | {_ratio_cell(closures['ratio'])} |"
                )
            lines.append(row)
        lines.append("")

    coverage = result["coverage_change"]
    lines.append(f"## Coverage change: {coverage['old']} → {coverage['new']}")
    lines.append("")
    lines.append("| population | pairs old | pairs new | lat/lat old | lat/lat new | cyr-any old | cyr-any new | closures old | closures new | grounded-or-looked-up closures old | new | lookup pairs also grounded old | new |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for population, block in coverage["populations"].items():
        lines.append(
            f"| {population} | {_cell(block['pairs']['old'])} | {_cell(block['pairs']['new'])} "
            f"| {_cell(block['pairs_lat_lat']['old'])} | {_cell(block['pairs_lat_lat']['new'])} "
            f"| {_cell(block['pairs_cyr_any']['old'])} | {_cell(block['pairs_cyr_any']['new'])} "
            f"| {_cell(block['closures']['old'])} | {_cell(block['closures']['new'])} "
            f"| {_cell(block['closures_grounded_or_looked_up']['old'])} | {_cell(block['closures_grounded_or_looked_up']['new'])} "
            f"| {block['lookup_pairs_also_grounded']['old']} | {block['lookup_pairs_also_grounded']['new']} |"
        )
    lines.append("")
    anchors = result["anchors"]
    lines.append(f"## Query anchors: edges the live path would write, {anchors['old']} → {anchors['new']}")
    lines.append("")
    lines.append(
        "The grounding verdict that decides credit also decides the anchor: a closure with at "
        "least one grounded result writes its query as an anchor with one edge per grounded "
        "result (before dedup against an existing anchor for the same query)."
    )
    lines.append("")
    lines.append("| population | closures | anchor-writing closures old | new | edges old | new | edges per closure old | new | edges per anchor-writing closure old | new |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for population, block in anchors["populations"].items():
        o, n = block["old"], block["new"]
        lines.append(
            f"| {population} | {o['closures']} | {o['closures_writing_an_anchor']} | {n['closures_writing_an_anchor']} "
            f"| {o['anchor_edges']} | {n['anchor_edges']} | {o['edges_per_closure']:.3f} | {n['edges_per_closure']:.3f} "
            f"| {o['edges_per_anchor_writing_closure']:.3f} | {n['edges_per_anchor_writing_closure']:.3f} |"
        )
    lines.append("")

    holdout = result.get("holdout")
    lines.append("## Holdout replay: `credit_rule_ab.py` on the sfx snapshot, old vs new threshold")
    lines.append("")
    if not holdout:
        lines.append("Not run (`--skip-holdout`).")
        lines.append("")
    else:
        lines.append(
            f"DB `{holdout['db']}` (read-only), grounded arm vs incumbent `{INCUMBENT_ARM}`; thresholds "
            f"enter through `{MIN_CONTAINMENT_ENV_VAR}`. Under the `grounded` label the label itself "
            "moves with the threshold, so compare the grounded arm with the incumbent *within* a "
            "run (paired bootstrap 95% CI over holdout events); under the `reconsumed`/`usefulness` "
            "labels the label is threshold-independent, so the grounded arm's figures at the two "
            "thresholds are directly comparable (point deltas; no cross-run bootstrap is available "
            "without editing `credit_rule_ab.py`) and the incumbent must be identical."
        )
        lines.append("")
        for cutoff, block in holdout["summary"]["cutoffs"].items():
            lines.append(f"### Cutoff {cutoff} ({block['usable_events']} usable events)")
            lines.append("")
            lines.append("| threshold | grounded results / graded | grounded share | per-event vs corpus IDF agreement | grounded-arm weight updates | mean graph | max graph | scopes graph > 0.5 | winner | grounded arm passes |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|")
            for label in ("old", "new"):
                if label not in block["grounding"]:
                    continue
                g = block["grounding"][label]
                s = block["weight_sanity"][label]
                v = block["verdict"][label]
                updates = block["labels"][PRIMARY_LABEL][label][GROUNDED_ARM]["weight_updates"]
                lines.append(
                    f"| {label} ({holdout['summary'][label]}) | {g['grounded_results']}/{g['graded_results']} "
                    f"| {g['grounded_share']:.4f} | {g['corpus_idf_agreement']:.4f} | {updates} "
                    f"| {s['mean_graph']:.4f} | {s['max_graph']:.4f} | {s['graph_over_limit']} "
                    f"| {v['winner']} | {'yes' if v['grounded_arm']['pass'] else 'no'} |"
                )
            lines.append("")
            lines.append("| label | threshold | events w/ useful | incumbent hit@5 | grounded hit@5 | Δ hit@5 [CI95] | incumbent MRR | grounded MRR | Δ MRR [CI95] |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for label_protocol in (PRIMARY_LABEL, *CONTROL_LABELS):
                rows = block["labels"][label_protocol]
                for label in ("old", "new"):
                    if label not in rows:
                        continue
                    r = rows[label]
                    d = r["grounded_vs_incumbent"]
                    lines.append(
                        f"| {label_protocol} | {label} ({holdout['summary'][label]}) | {r[GROUNDED_ARM]['events_with_useful']} "
                        f"| {r[INCUMBENT_ARM]['hit@5']:.4f} | {r[GROUNDED_ARM]['hit@5']:.4f} "
                        f"| {d['hit@5']['delta']:+.4f} [{d['hit@5']['ci95'][0]:+.4f}, {d['hit@5']['ci95'][1]:+.4f}]"
                        f"{' REGRESSION' if d['hit@5']['regression'] else (' gain' if d['hit@5']['gain'] else '')} "
                        f"| {r[INCUMBENT_ARM]['mrr']:.4f} | {r[GROUNDED_ARM]['mrr']:.4f} "
                        f"| {d['mrr']['delta']:+.4f} [{d['mrr']['ci95'][0]:+.4f}, {d['mrr']['ci95'][1]:+.4f}]"
                        f"{' REGRESSION' if d['mrr']['regression'] else (' gain' if d['mrr']['gain'] else '')} |"
                    )
                if "new_minus_old" in rows:
                    g = rows["new_minus_old"]["grounded_arm"]
                    i = rows["new_minus_old"]["incumbent_arm"]
                    lines.append(
                        f"| {label_protocol} | new − old | | | | grounded arm Δ hit@5 {g['hit@5']:+.4f} (incumbent {i['hit@5']:+.4f}) "
                        f"| | | grounded arm Δ MRR {g['mrr']:+.4f} (incumbent {i['mrr']:+.4f}) |"
                    )
            lines.append("")
            for control, item in block["controls"].items():
                delta = item["grounded_arm_new_minus_old"]
                lines.append(
                    f"- `{control}` control: incumbent identical across thresholds = "
                    f"{item['incumbent_identical_across_thresholds']}; grounded arm new − old "
                    f"hit@5 {delta['hit@5']:+.4f} (≈ {item['grounded_arm_new_minus_old_hit5_events']:+d} "
                    f"of {item['events_with_useful']} events), MRR {delta['mrr']:+.4f} → "
                    f"{'**moves against the new threshold**' if item['moves_against_new'] else 'does not move against the new threshold'}; "
                    f"at the new threshold vs incumbent: hit@5 "
                    f"{'REGRESSION' if item['new_arm_vs_incumbent']['hit@5']['regression'] else ('gain' if item['new_arm_vs_incumbent']['hit@5']['gain'] else 'indistinguishable')}, "
                    f"MRR {'REGRESSION' if item['new_arm_vs_incumbent']['mrr']['regression'] else ('gain' if item['new_arm_vs_incumbent']['mrr']['gain'] else 'indistinguishable')}; "
                    f"significant regressions vs incumbent under this label at the old threshold: "
                    f"{_regressions(item['old_arm_vs_incumbent']) or 'none'}, at the new: "
                    f"{_regressions(item['new_arm_vs_incumbent']) or 'none'}."
                )
            lines.append(
                f"- Controls move against the new threshold at this cutoff: "
                f"**{'yes' if block['controls_move_against_new'] else 'no'}**; significant control "
                f"regression vs incumbent at the new threshold: "
                f"**{'yes' if block['controls_significant_regression_at_new'] else 'no'}**."
            )
            lines.append("")
    for note in result.get("notes", []):
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus-dir", default=str(CORPUS_DIR))
    parser.add_argument("--holdout-db", default=str(CORPUS_DIR / f"sfx-{SNAPSHOT_DATE}.sqlite3"))
    parser.add_argument("--holdout-cutoff", action="append", default=None)
    parser.add_argument("--holdout-dir", default=str(DEFAULT_HOLDOUT_DIR))
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument("--force-holdout", action="store_true", help="re-run credit_rule_ab even if a report exists")
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "--credit-ab-worker":
        return credit_ab_worker(raw[1:])
    args = build_parser().parse_args(raw)
    corpus_dir = Path(args.corpus_dir).expanduser()
    corpora: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for host in HOSTS:
        path = ensure_corpus(host, corpus_dir)
        loaded = usr.load_corpus([path])
        corpora.append(
            {
                "host": host,
                "path": str(path),
                "sha256": _sha256(path),
                "records": len(loaded),
                "pairs": sum(len(record["results"]) for record in loaded),
            }
        )
        records.extend(loaded)

    thresholds = sweep_thresholds()
    sweep = run_sweep(records, thresholds)
    adopted = sweep["adopted"] if sweep["adopted"] is not None else PREVIOUS_MIN_CONTAINMENT
    result: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkout": {"commit": _git("rev-parse", "--short", "HEAD"), "dirty": bool(_git("status", "--porcelain"))},
        "tokenizer": {
            "cyrillic_stem_enabled": cyrillic_stem_enabled(),
            "default_min_containment": DEFAULT_MIN_CONTAINMENT,
            "calibrated_min_containment": CALIBRATED_MIN_CONTAINMENT,
        },
        "corpora": corpora,
        "criterion": {
            "noise_ceiling": NOISE_CEILING,
            "snr_floor": SNR_FLOOR,
            "min_class_pairs": MIN_CLASS_PAIRS,
            "cuts": [{"related": r, "unrelated": u} for r, u in CUTS],
            "populations": list(POPULATIONS),
            "rule": "lowest threshold such that it and every higher threshold in the sweep clear "
            "noise <= ceiling and ratio >= floor on every population at both cuts",
        },
        "previous": PREVIOUS_MIN_CONTAINMENT,
        "adopted": adopted,
        "adopted_by_criterion": sweep["adopted"],
        "candidate_0_15": {
            "threshold": CANDIDATE_MIN_CONTAINMENT,
            "clears": bool(_entry(sweep, CANDIDATE_MIN_CONTAINMENT) and _entry(sweep, CANDIDATE_MIN_CONTAINMENT)["clears"]),
            "reasons": explain_failure(sweep, CANDIDATE_MIN_CONTAINMENT),
        },
        "sweep": sweep,
        "coverage_change": coverage_change(sweep, PREVIOUS_MIN_CONTAINMENT, adopted),
        "anchors": anchor_change(sweep, PREVIOUS_MIN_CONTAINMENT, adopted),
        "notes": [],
    }
    if CALIBRATED_MIN_CONTAINMENT != adopted:
        result["notes"].append(
            f"The checkout ships CALIBRATED_MIN_CONTAINMENT = {CALIBRATED_MIN_CONTAINMENT}, "
            f"which differs from the adopted {adopted}; grounding.py must be updated."
        )

    if not args.skip_holdout:
        cutoffs = tuple(args.holdout_cutoff) if args.holdout_cutoff else DEFAULT_HOLDOUT_CUTOFFS
        db = Path(args.holdout_db).expanduser()
        holdout_dir = Path(args.holdout_dir)
        reports: dict[str, dict[str, dict[str, Any]]] = {}
        for cutoff in cutoffs:
            reports[cutoff] = {}
            for label, threshold in (("old", PREVIOUS_MIN_CONTAINMENT), ("new", adopted)):
                if label == "new" and threshold == PREVIOUS_MIN_CONTAINMENT:
                    continue
                reports[cutoff][label] = run_holdout(
                    db, cutoff, threshold, holdout_dir, force=args.force_holdout
                )
        result["holdout"] = {
            "db": str(db),
            "cutoffs": list(cutoffs),
            "reports": {
                cutoff: {
                    label: str(holdout_report_path(holdout_dir, PREVIOUS_MIN_CONTAINMENT if label == "old" else adopted, cutoff))
                    for label in by_threshold
                }
                for cutoff, by_threshold in reports.items()
            },
            "summary": summarize_holdout(reports, PREVIOUS_MIN_CONTAINMENT, adopted),
        }

    notes = result["notes"]
    notes.append(
        "Holdout evidence gate: `replay.snapshot_evidence` in this checkout asks `nodes.embedding`, "
        "which the 2026-09 snapshots dropped for `node_chunk_embeddings`; the A/B was run through "
        "`grounding_recalibration.credit_ab_worker`, which substitutes a gate that mirrors "
        "`MemoryStore._has_vector_evidence` when the column is absent and leaves `credit_rule_ab.py` "
        "and `replay.py` untouched (the lookup-credit branch carries the same fix inside replay.py)."
    )
    notes.append(
        "Attestation coupling: `attestation.ATTESTATION_MIN_CONTAINMENT` binds "
        "`feedback.RECALL_CREDIT_MIN_CONTAINMENT`, so post-session attestation now grades at the "
        "adopted value too; the shuffled-session noise bounds in docs/post-session-attestation.md "
        "were measured at 0.25 and have not been re-measured."
    )
    holdout_summary = result.get("holdout", {}).get("summary")
    if holdout_summary:
        for cutoff, block in holdout_summary["cutoffs"].items():
            for control, item in block["controls"].items():
                delta = item["grounded_arm_new_minus_old"]
                if item["moves_against_new"]:
                    notes.append(
                        f"Cutoff {cutoff}, `{control}` control moves against {adopted}: grounded-arm "
                        f"holdout hit@5 {delta['hit@5']:+.4f} "
                        f"(≈ {item['grounded_arm_new_minus_old_hit5_events']:+d} of "
                        f"{item['events_with_useful']} events), MRR {delta['mrr']:+.4f}."
                    )
                new_regr = _regressions(item["new_arm_vs_incumbent"])
                old_regr = _regressions(item["old_arm_vs_incumbent"])
                if new_regr:
                    notes.append(
                        f"Cutoff {cutoff}, `{control}` label: the grounded arm regresses significantly "
                        f"against `{INCUMBENT_ARM}` on {new_regr} at {adopted}; at {PREVIOUS_MIN_CONTAINMENT} "
                        f"the same regression is {'present (' + old_regr + ')' if old_regr else 'absent'}, "
                        "so it belongs to the grounded credit policy on this snapshot rather than to the "
                        "threshold."
                    )

    # The raw replay rows are what the checks and coverage blocks were computed
    # from; they are not repeated in the artifact (``per_threshold`` carries the
    # figures the criterion reads, ``coverage_change``/``anchors`` the rest).
    result["sweep"] = {
        **sweep,
        "cuts": {
            key: {name: value for name, value in block.items() if name != "rows"}
            for key, block in sweep["cuts"].items()
        },
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = render_markdown(result)
    Path(args.out_md).write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"adopted={adopted} -> {out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
