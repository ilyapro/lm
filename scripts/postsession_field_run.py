#!/usr/bin/env python3
"""The sealed-holdout field measurement for the post-session extraction stage.

This is the run that decides whether the extraction stage may be scaled, and it
is deliberately split into inspectable phases so the pre-registration provably
precedes every holdout number::

    scripts/postsession_field_run.py seal-check    # reconstruct + prove the seal
    scripts/postsession_field_run.py freeze        # SQLite-backup snapshot of the live DB
    scripts/postsession_field_run.py preregister   # freeze the plan BEFORE any number
    scripts/postsession_field_run.py extract --split eval
    scripts/postsession_field_run.py extract --split holdout
    scripts/postsession_field_run.py control       # organic control, identical protocol
    scripts/postsession_field_run.py seed-and-measure --split eval
    scripts/postsession_field_run.py seed-and-measure --split holdout
    scripts/postsession_field_run.py audit --split eval
    scripts/postsession_field_run.py audit --split holdout
    scripts/postsession_field_run.py verdict
    scripts/postsession_field_run.py report

Every phase writes its state under ``--state-dir`` (default
``tmp/field-<batch>``, gitignored: dry-run reports and op dumps carry session
content and must never be tracked). The only tracked outputs are
``artifacts/post-session/field-report.json`` / ``field-report.md``, which carry
aggregates, digests and node ids only and are passed through the same privacy
guard the baseline artifact used.

Protocol facts this driver relies on (verified against the modules it imports):

* The pre-registered bar lives in ``artifacts/post-session/usage-baseline.json``
  and is enforced by ``runner.evaluate_gate``; the pinned protocol is
  ``counterfactual_post_cutoff`` (absolute floor 0.05 — the deprecated scalar
  0.15 is never read, exactly as the artifact instructs), the relative floor is
  0.5 x the concurrently measured organic control, and the control must itself
  be admissible.
* Extracted and organic cohorts are measured by the SAME tool
  (``scripts/counterfactual_consumption.py``) with byte-identical protocol
  fields: ``--as-of``, ``--replay-since``, ``min_containment``, ``within_days``.
* A candidate trace has no history and no chunk embeddings; the counterfactual
  harness embeds it via the ordinary lazy drain on the read path, which is the
  documented path for judging candidate cohorts.
* ``CassetteJudge`` recording is not safe across concurrent instances sharing a
  path (whole-file rewrite, last writer wins); :class:`LockingCassetteJudge`
  below serializes cassette I/O while keeping the paid CLI calls parallel.

The sealed split is read exactly once per phase, on purpose, and only here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from living_memory.config import MemoryConfig  # noqa: E402
from living_memory.consolidation import memory_teach  # noqa: E402
from living_memory.postsession.corpus import (  # noqa: E402
    DEFAULT_MANIFEST,
    SessionEntry,
    assign_splits,
    build_corpus,
    index_path_for,
    iter_index,
    load_session,
    read_manifest,
    roots_from_manifest,
    split_digest,
    write_index,
)
from living_memory.postsession.insights import ExtractionConfig  # noqa: E402
from living_memory.postsession.judge import (  # noqa: E402
    CassetteJudge,
    ClaudeCliJudge,
    JudgeError,
    canonical_json,
    default_redactor,
)
from living_memory.postsession.runner import (  # noqa: E402
    DEFAULT_BASELINE,
    DEFAULT_BOOTSTRAP_ALLOWANCE,
    DEFAULT_STATE_DIR,
    DEFAULT_VERDICT,
    ExtractionRunner,
    Ledger,
    RunnerConfig,
    build_report,
    enforce_attribution,
    evaluate_gate,
    op_fingerprint,
)
from living_memory.postsession.usage_metric import check_privacy  # noqa: E402
from living_memory.retrieval_harness import backup_database, create_snapshot  # noqa: E402
from living_memory.storage import MemoryStore  # noqa: E402

# ---------------------------------------------------------------------------
# THE PRE-REGISTERED PLAN. Constants only: nothing below reads a measured
# holdout number. This block is committed to git before the holdout phases run,
# and `preregister` publishes it (with its sha256) before `extract --split
# holdout` will agree to start.
# ---------------------------------------------------------------------------

BATCH = "field-20260819"

#: Pinned from the published baseline: cuts BOTH nodes and recall_events, so
#: the run is reproducible although the live DB keeps growing.
AS_OF = "2026-08-19T00:00:00Z"
#: The pinned replay arm — the only one that cleared the harness's own
#: 10-point self-validation (usage-baseline.json $.self_check).
REPLAY_SINCE = "2026-08-15T00:00:00Z"
#: Every seeded trace is stamped with this created_at: one instant just before
#: the replay cutoff. Matches the bar's own framing ("written at one instant,
#: consumed only by traffic recorded later") and gives the extracted cohort the
#: same recency posture as the organic control (whose members are 0-3 days old
#: at the cutoff), so recency is not a confounder in the comparison.
SEED_CREATED_AT = "2026-08-14T23:59:59Z"
#: One selector window for every cohort: the organic control's published window.
#: All seeded nodes carry SEED_CREATED_AT, which lies inside it; membership is
#: then distinguished by the context marker alone.
COHORT_WINDOW = "2026-08-12..2026-08-15"
#: Published organic-control cohort rule, verbatim from usage-baseline.json.
CONTROL_COHORT_RULE = (
    "nodes.created_at in [2026-08-12,2026-08-15) UTC at as_of 2026-08-19T00:00:00Z; "
    "all levels, decayed included, no agent/context filter; disjoint from every "
    "published baseline cohort"
)

#: Session eligibility: the session must have FINISHED before the replay
#: cutoff, so its extracted traces could really have existed before every
#: replayed event. ae_node_result transcripts carry no probed timestamps, so
#: file mtime is the pre-registered fallback (a node result is written once,
#: at completion).
SESSION_END_BEFORE = REPLAY_SINCE
SESSIONS_PER_SPLIT = 300
MAX_SESSION_BYTES = 12_000_000
#: A verdict computed over fewer seeded traces than this is INCONCLUSIVE:
#: the extractor proposes ~0.07-0.1 node-writing ops per session (its anti-dump
#: gate is deliberately harsh), so a small session sample cannot separate the
#: relative floor from noise. Registered after observing EVAL yield only; no
#: holdout session had been read.
MIN_EXTRACTED_COHORT = 20

#: Runner budgets for the measurement dry-runs. Wide enough that the total-op
#: budget never binds (the cohort should be what the extractor yields, not what
#: a cap leaves), per-session cap at the runner default.
MAX_OPS_PER_SESSION = 3
MAX_TOTAL_OPS = 90
EXTRACT_JUDGE_MODEL = "sonnet"
EXTRACT_WORKERS = 4

#: Adversarial anti-dump audit: refute-first, content-only, a model tier above
#: the extraction judge so the auditor is not the same distribution grading
#: itself. Sample is deterministic (ops sorted by fingerprint).
AUDIT_SAMPLE_PER_SPLIT = 20
AUDIT_JUDGE_MODEL = "opus"
AUDIT_TIMEOUT_S = 240.0
AUDIT_PRECISION_BAR = 0.70
#: If more than this share of the audit sample errors out, the audit is
#: INCONCLUSIVE and the overall verdict cannot PASS.
AUDIT_MAX_ERROR_SHARE = 0.20
AUDIT_TASK = (
    "You are an adversarial auditor for an automated memory extractor. A proposed "
    "memory trace is given, extracted offline from a finished agent session "
    "transcript. Your default verdict is 'dump'. Return 'genuine' ONLY if the "
    "trace states a specific, verifiable fact that would be expensive to re-derive "
    "from the code or git history alone (a resolved failure and its cause, a "
    "measured result, a refuted hypothesis, an external contract, a non-obvious "
    "pitfall), names its subject concretely (paths, identifiers, numbers), and is "
    "one fact rather than a bundle. Return 'dump' if it is a work journal or "
    "completion note (did X, finished Y), a restatement of something any reader of "
    "the diff or repository would see, vague advice, several facts bundled into "
    "one trace, or speculation. Judge only what is written in the trace."
)
AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["genuine", "dump"]},
        "reason": {"type": "string", "minLength": 1, "maxLength": 400},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}

#: The eval->holdout generalization bar applies to the adversarial audit
#: precision (identical audit protocol and sample rule on both splits):
#: relative degradation (eval - holdout) / eval must be <= 0.25. The
#: counterfactual consumption degradation is reported alongside as a
#: diagnostic; consumption on holdout is already tested directly against the
#: organic control, and the small extracted cohorts make its eval/holdout
#: ratio statistically much weaker than the equal-n audit comparison.
DEGRADATION_BAR = 0.25

#: Live-write cap on PASS, through the real runner CLI.
LIVE_WRITE_CAP = 25

PREREGISTRATION: dict[str, Any] = {
    "artifact": "field-measurement-preregistration",
    "batch": BATCH,
    "registered_by": "scripts/postsession_field_run.py",
    "bar_source": "artifacts/post-session/usage-baseline.json#pre_registered_bar",
    "bar_notes": [
        "The binding bar is the baseline's own decision procedure, applied by "
        "runner.evaluate_gate:",
        "PASS iff the concurrently measured organic control is admissible AND "
        "extracted_rate >= 0.5 * organic_rate AND",
        "extracted_rate >= absolute_floor_by_protocol[counterfactual_post_cutoff] "
        "= 0.05.",
        "The goal text's scalar '>= 0.15 absolute' is the deprecated "
        "retrospective_long_window floor.",
        "The baseline amended it before any extracted cohort existed: under the "
        "pinned replay arm a 0.15 floor",
        "would reject the known-good organic control itself (0.1159). Both "
        "comparisons are reported;",
        "the protocol-keyed floor is the binding one.",
    ],
    "protocol": {
        "as_of": AS_OF,
        "replay_since": REPLAY_SINCE,
        "cohort_window": COHORT_WINDOW,
        "seed_created_at": SEED_CREATED_AT,
        "reinsert_mode_extracted": "fresh",
        "reinsert_mode_control": "faithful",
        "tool": "scripts/counterfactual_consumption.py",
        "identical_for_both_cohorts": True,
        "seed_marker_context_key": "field_batch",
    },
    "seal_strategy": [
        "verify holdout_sha256 by full re-derivation, else by a witness search "
        "over mtime-undecided entries;",
        "if the digest is unrecoverable because AE archival renamed path-derived "
        "ae_node_result keys",
        "or the CLI rolling window pruned the oldest transcript sliver, fall "
        "back to a per-session membership proof:",
        "stable-key source, transcript-internal started_at before the sealing "
        "instant, singleton identity group",
        "(split_key == session_key, so the bucket is a pure function of the "
        "key), own-key bucket in the sealed split's buckets,",
        "and absence from every session key named by tracked development "
        "artifacts. A session failing the proof is never read.",
    ],
    "session_rule": {
        "eligible": [
            "corpus split member satisfying the per-session sealed-membership "
            "proof (see seal_strategy),",
            f"whose session ended before the replay cutoff {SESSION_END_BEFORE} "
            "(transcript-internal ended_at),",
            f"and whose transcript is at most {MAX_SESSION_BYTES} bytes",
        ],
        "order": "sha256(session_key), ascending (the runner's own sample rank)",
        "sessions_per_split": SESSIONS_PER_SPLIT,
        "train_set": "train split (used by sibling children during development)",
        "eval_set": f"first {SESSIONS_PER_SPLIT} eligible eval-split sessions",
        "holdout_set": (
            f"first {SESSIONS_PER_SPLIT} eligible sealed holdout-split sessions; "
            "read only by this measurement, seal proven first"
        ),
    },
    "extraction": {
        "stages": ["insights", "corrections", "attest"],
        "judge": f"claude-cli {EXTRACT_JUDGE_MODEL} via recording cassette",
        "max_ops_per_session": MAX_OPS_PER_SESSION,
        "max_total_ops": MAX_TOTAL_OPS,
        "dry_run": True,
        "lm_dedup": False,
        "measured_cohort": "ops whose dry-run decision is would_write (remember+teach)",
    },
    "audit": {
        "sample_per_split": AUDIT_SAMPLE_PER_SPLIT,
        "sample_order": "ops sorted by op fingerprint, first N",
        "judge_model": AUDIT_JUDGE_MODEL,
        "payload": "kind + trace text only (no evidence, no session context)",
        "precision": "genuine / (genuine + dump); judge errors excluded from the "
        "denominator, reported, and > 20% errors makes the audit INCONCLUSIVE",
        "precision_bar": AUDIT_PRECISION_BAR,
        "task_sha256": hashlib.sha256(AUDIT_TASK.encode("utf-8")).hexdigest(),
    },
    "generalization": {
        "metric": "adversarial audit precision, eval -> holdout",
        "definition": "(precision_eval - precision_holdout) / precision_eval",
        "bar": DEGRADATION_BAR,
        "diagnostic": "counterfactual consumption rate degradation eval -> holdout "
        "is reported alongside, not gated (unequal, small cohorts)",
    },
    "verdict_rule": (
        "PASS iff evaluate_gate(extracted vs concurrent organic control) is PASS "
        f"AND holdout audit precision >= {AUDIT_PRECISION_BAR} AND audit precision "
        f"degradation <= {DEGRADATION_BAR} AND the seeded holdout cohort holds >= "
        f"{MIN_EXTRACTED_COHORT} traces; INCONCLUSIVE anywhere means not PASS"
    ),
    "amendments": [
        "v2 (before extract --split holdout): seal argument moved from digest "
        "re-derivation to a per-session membership proof",
        "(AE archival renames path-derived ae_node_result keys; the CLI rolling "
        "window pruned the 2026-07-20 morning sliver).",
        "v3 (before extract --split holdout; after observing EVAL yield only): "
        "sessions_per_split raised 30 -> 300 and",
        f"min_extracted_cohort {MIN_EXTRACTED_COHORT} added, because the "
        "extractor's anti-dump gate is so selective",
        "(49 proposals per 747 eval sessions in insights-eval.json) that 30 "
        "sessions cannot yield a decision-grade cohort.",
        "No bar, floor, cutoff, replay set or audit rule was changed by any "
        "amendment.",
    ],
    "on_pass": f"at most {LIVE_WRITE_CAP} holdout-derived traces written through "
    "scripts/post_session_extract.py --write, node ids and memory_forget "
    "reversal commands recorded in the field report",
    "on_fail": "zero live writes; the report states what would have to change",
}

FIELD_REPORT_JSON = REPO_ROOT / "artifacts/post-session/field-report.json"
FIELD_REPORT_MD = REPO_ROOT / "artifacts/post-session/field-report.md"
LIVE_DB = Path("~/.local/share/living-memory/global.sqlite3").expanduser()

_SOURCES_WITHOUT_TIMESTAMPS = ("ae_node_result",)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(path: Path, hint: str) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing {path}: run `{hint}` first")
    return read_json(path)


# ---------------------------------------------------------------------------
# A cassette judge that is safe to record into from concurrent workers.
# ---------------------------------------------------------------------------


class LockingCassetteJudge(CassetteJudge):
    """CassetteJudge with merge-on-save under a class-wide lock.

    The parent rewrites the whole cassette file from one instance's private
    entry dict; with a fresh instance per stage per session (the runner's
    factory contract) and parallel workers, recordings race and the last writer
    wins. Here every cache read and write first merges the file's current
    entries, and only the cassette I/O is serialized — the paid CLI call itself
    stays parallel.
    """

    _io_lock = threading.Lock()

    def judge(
        self,
        task: str,
        schema: dict[str, Any],
        payload: dict[str, Any],
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        envelope = self._envelope(task, schema, payload)
        key = self.key_for(envelope, self.model)
        with self._io_lock:
            if key not in self._entries and self.path.exists():
                self._entries.update(self._load())
            entry = self._entries.get(key)
        if entry is not None:
            return self._replay(entry, envelope, schema)
        if not self.record or self.inner is None:
            return super().judge(task, schema, payload, timeout_s=timeout_s)

        stored: dict[str, Any] = {
            "task": envelope.task,
            "model": self.model,
            "schema_sha256": envelope.schema_sha256,
            "system_sha256": envelope.system_sha256,
            "prompt_sha256": envelope.prompt_sha256,
            "prompt": envelope.user_prompt,
            "recorded_at": self._now(),
        }
        try:
            response = self.inner.judge(task, schema, payload, timeout_s=timeout_s)
        except JudgeError as exc:
            from living_memory.postsession.judge import (
                JudgeInvalidOutput,
                JudgeRefused,
                JudgeStats,
            )

            if not isinstance(exc, (JudgeRefused, JudgeInvalidOutput)):
                raise
            stored["outcome"] = "refused" if isinstance(exc, JudgeRefused) else "invalid"
            stored["error"] = str(exc)
            stored["stats"] = self._inner_stats(envelope, stored["outcome"])
            self._store(key, stored)
            self._record(JudgeStats.from_dict(stored["stats"]))
            raise
        from living_memory.postsession.judge import JudgeStats

        stored["outcome"] = "ok"
        stored["response"] = response
        stored["stats"] = self._inner_stats(envelope, "ok")
        self._store(key, stored)
        self._record(JudgeStats.from_dict(stored["stats"]))
        return response

    def _store(self, key: str, stored: dict[str, Any]) -> None:
        with self._io_lock:
            if self.path.exists():
                self._entries.update(self._load())
            self._entries[key] = stored
            self._save()


# ---------------------------------------------------------------------------
# seal-check: reconstruct the sealed corpus world and prove the seal unmoved
# ---------------------------------------------------------------------------


def _entry_instant(entry: SessionEntry) -> str:
    """When this session's transcript came to exist, best evidence first."""

    if entry.started_at:
        return str(entry.started_at)
    if entry.ended_at:
        return str(entry.ended_at)
    try:
        stat = Path(entry.path).stat()
    except OSError:
        return "9999-12-31T00:00:00Z"
    return (
        datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _entry_end(entry: SessionEntry) -> str | None:
    """When the session finished; mtime fallback for un-timestamped sources."""

    if entry.ended_at:
        return str(entry.ended_at)
    if entry.source in _SOURCES_WITHOUT_TIMESTAMPS:
        try:
            stat = Path(entry.path).stat()
        except OSError:
            return None
        return (
            datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    return None


def cmd_seal_check(args: argparse.Namespace) -> int:
    """Reconstruct the sealed corpus world and prove every split digest.

    Membership evidence, strongest first: a transcript-internal timestamp
    (started_at/ended_at) on either side of ``manifest.generated_at`` is
    decisive; a file whose only dating evidence is its mtime *after* the
    sealing instant is undecided — result files get their mtimes bumped by
    tooling, so a late mtime does not prove a late file. The undecided pool is
    tiny, so the sealed set is recovered exactly by a per-split witness
    search: choose the members that make the recomputed split digest equal
    the sealed one. The digest is the proof; the search only finds the
    witness. Split assignment itself is a stable pure function (verified: the
    subset and full-disk derivations agree on every shared session).
    """

    from itertools import combinations

    state = Path(args.state_dir)
    manifest = read_manifest(DEFAULT_MANIFEST)
    generated_at = str(manifest["generated_at"])
    sealed_digests = {name: str(manifest["split_sha256"][name]) for name in
                      ("train", "eval", "holdout")}
    sealed_counts = manifest["counts"]["by_split"]
    log(f"rehashing every transcript under the manifest's roots (sealed {generated_at})…")
    started = time.monotonic()
    corpus = build_corpus(roots_from_manifest(manifest), sources=manifest.get("sources"))
    log(f"  {len(corpus.entries)} transcripts hashed in {time.monotonic() - started:.0f}s")

    full_match = corpus.manifest["split_sha256"] == sealed_digests

    certain_in: list[SessionEntry] = []
    certain_out: list[SessionEntry] = []
    undecided: list[SessionEntry] = []
    for entry in corpus.entries:
        stamp = entry.started_at or entry.ended_at
        if stamp:
            (certain_in if str(stamp) <= generated_at else certain_out).append(entry)
        elif _entry_instant(entry) <= generated_at:
            certain_in.append(entry)
        else:
            undecided.append(entry)

    witness: dict[str, Any] = {"pool": len(undecided), "chosen": [], "matched": {}}
    all_match = True
    for name in ("train", "eval", "holdout"):
        base = sorted(e.session_key for e in certain_in if e.split == name)
        candidates = sorted(e.session_key for e in undecided if e.split == name)
        need = int(sealed_counts[name]) - len(base)
        found = None
        if 0 <= need <= len(candidates) and len(candidates) - need <= 3:
            for combo in combinations(range(len(candidates)), len(candidates) - need):
                chosen = [candidates[i] for i in range(len(candidates)) if i not in combo]
                if split_digest(base + chosen) == sealed_digests[name]:
                    found = [candidates[i] for i in combo]
                    break
        witness["matched"][name] = found is not None
        all_match = all_match and found is not None
        log(f"  seal[{name}]: {'MATCH' if found is not None else 'no witness'}")

    # Digest re-derivation can fail for reasons that move no session between
    # splits: ae_node_result identity is path-derived and AE archival renames
    # paths, and the Claude CLI's rolling cleanup prunes the oldest sliver.
    # Quantify both, then fall back to a per-session membership proof.
    sealed_by_source: dict[str, int] = {}
    for split_sources in manifest["counts"]["by_split_source"].values():
        for source, count in split_sources.items():
            sealed_by_source[source] = sealed_by_source.get(source, 0) + int(count)
    now_certain_by_source: dict[str, int] = {}
    for entry in certain_in:
        now_certain_by_source[entry.source] = now_certain_by_source.get(entry.source, 0) + 1
    reconciliation = {}
    stable_ok = True
    for source in sorted(set(sealed_by_source) | set(now_certain_by_source)):
        sealed_n = sealed_by_source.get(source, 0)
        certain_n = now_certain_by_source.get(source, 0)
        undecided_n = sum(1 for e in undecided if e.source == source)
        missing = max(0, sealed_n - certain_n - undecided_n)
        reconciliation[source] = {
            "sealed": sealed_n,
            "on_disk_pre_sealing": certain_n,
            "mtime_undecided": undecided_n,
            "missing_lower_bound": missing,
        }
        if source not in ("ae_node_result", "claude") and missing:
            stable_ok = False
    claude_missing = reconciliation.get("claude", {}).get("missing_lower_bound", 0)
    renamed_evidence = sum(1 for e in undecided if "/archive/" in e.session_key)
    membership_proof_available = bool(stable_ok and claude_missing <= 12)

    dev_keys = _dev_artifact_keys()
    index_path = index_path_for(Path(DEFAULT_MANIFEST))
    stats = write_index(corpus.entries, index_path)
    write_json(state / "dev-session-keys.json", {"keys": sorted(dev_keys)})
    result = {
        "artifact": "seal-check",
        "checked_at": utc_now(),
        "manifest_generated_at": generated_at,
        "sealed_holdout_sha256": sealed_digests["holdout"],
        "disk_now": {
            "transcripts": len(corpus.entries),
            "matches_seal": full_match,
            "certain_in": len(certain_in),
            "certain_out": len(certain_out),
            "undecided_mtime_only": len(undecided),
        },
        "witness_search": witness,
        "digest_rederivable": bool(full_match or all_match),
        "why_not_rederivable": [
            "ae_node_result session keys are path-derived and AE archival renames "
            "node paths post-sealing",
            f"({renamed_evidence} undecided keys carry /archive/ segments; every "
            "result.md mtime was touched after sealing),",
            "and the Claude CLI rolling cleanup pruned the oldest sliver of "
            f"claude transcripts (lower bound {claude_missing};",
            "earliest survivor started 2026-07-20T10:26Z). Neither mechanism "
            "moves a surviving stable-key session between splits.",
        ]
        if not (full_match or all_match)
        else [],
        "per_source_reconciliation": reconciliation,
        "membership_proof": {
            "available": membership_proof_available,
            "rule": [
                "each measured session must individually satisfy:",
                "source with UUID/id-stable keys (never ae_node_result),",
                "transcript-internal started_at earlier than the sealing instant,",
                "singleton identity group with split_key == session_key (bucket is "
                "then a pure function of the key),",
                "sha256-bucket of its own key inside the split's sealed buckets,",
                "and absence from every session key named by tracked development "
                "artifacts",
            ],
        },
        "seal_unmoved": bool(full_match or all_match),
        "index_written": {
            "path": str(index_path),
            "entries": stats.lines,
            "sha256": stats.sha256,
            "note": "today's full derivation; selection applies the membership proof",
        },
    }
    write_json(state / "seal-check.json", result)
    if result["seal_unmoved"]:
        log("seal UNMOVED (digest re-derived)")
        return 0
    if membership_proof_available:
        log(
            "seal digest NOT re-derivable (renamed ae_node paths + pruned claude "
            f"sliver <= {claude_missing}); per-session membership proof is available"
        )
        return 0
    log("seal MOVED and no membership proof possible")
    return 1


def _dev_artifact_keys() -> set[str]:
    """Every session-key-shaped string in tracked development artifacts.

    A measured holdout session must not appear here: these artifacts were
    produced while the extractor was being built, so anything they name was
    readable during development.
    """

    import re

    keys: set[str] = set()
    pattern = re.compile(
        r'"((?:claude|codex|gigacode|deepseek|ae_chat|ae_node)[:][^"]{4,180})"'
    )
    for path in (REPO_ROOT / "artifacts/post-session").glob("*.json"):
        if path.name in ("field-report.json",):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for match in pattern.finditer(text):
            keys.add(match.group(1))
    return keys


# ---------------------------------------------------------------------------
# freeze + preregister
# ---------------------------------------------------------------------------


def cmd_freeze(args: argparse.Namespace) -> int:
    state = Path(args.state_dir)
    target = state / "snapshot.sqlite3"
    state.mkdir(parents=True, exist_ok=True)
    log(f"freezing {LIVE_DB} -> {target} (SQLite backup API; the WAL is half the data)…")
    started = time.monotonic()
    manifest = create_snapshot(LIVE_DB, target)
    manifest["frozen_in_seconds"] = round(time.monotonic() - started, 1)
    manifest["as_of_cut"] = AS_OF
    write_json(state / "snapshot-manifest.json", manifest)
    log(
        f"  sha256={manifest['snapshot_sha256']} rows={manifest['row_counts']} "
        f"in {manifest['frozen_in_seconds']}s"
    )
    return 0


def cmd_preregister(args: argparse.Namespace) -> int:
    state = Path(args.state_dir)
    snapshot = require(state / "snapshot-manifest.json", "freeze")
    seal = require(state / "seal-check.json", "seal-check")
    if not (
        seal.get("seal_unmoved")
        or (seal.get("membership_proof") or {}).get("available")
    ):
        raise SystemExit("neither seal digest nor membership proof holds; refusing")
    body = dict(PREREGISTRATION)
    body["registered_at"] = utc_now()
    body["snapshot_sha256"] = snapshot["snapshot_sha256"]
    body["snapshot_row_counts"] = snapshot["row_counts"]
    body["sealed_holdout_sha256"] = seal["sealed_holdout_sha256"]
    body["plan_sha256"] = hashlib.sha256(
        canonical_json(PREREGISTRATION).encode("utf-8")
    ).hexdigest()
    write_json(state / "preregistration.json", body)
    log(f"pre-registered plan {body['plan_sha256'][:16]}… at {body['registered_at']}")
    return 0


# ---------------------------------------------------------------------------
# extract: the full stage, dry-run, in-process (the CLI's own components)
# ---------------------------------------------------------------------------


def _select_sessions(
    split: str, state: Path
) -> tuple[list[SessionEntry], dict[str, Any]]:
    index_path = index_path_for(Path(DEFAULT_MANIFEST))
    if not index_path.is_file():
        raise SystemExit(f"no corpus index at {index_path}; run seal-check first")
    manifest = read_manifest(DEFAULT_MANIFEST)
    generated_at = str(manifest["generated_at"])
    buckets = set(int(b) for b in manifest["split_policy"][split])
    dev_keys = set(read_json(state / "dev-session-keys.json")["keys"])
    from living_memory.postsession.corpus import bucket_for

    pool = [e for e in iter_index(index_path) if e.split == split]
    eligible: list[SessionEntry] = []
    excluded = {
        "unstable_identity_source": 0,
        "not_provably_pre_sealing": 0,
        "grouped_identity": 0,
        "bucket_mismatch": 0,
        "named_by_dev_artifacts": 0,
        "no_end_timestamp": 0,
        "ended_after_cutoff": 0,
        "too_large": 0,
    }
    for entry in pool:
        # Per-session sealed-membership proof (see seal-check.json): the split
        # of a singleton, stable-key, pre-sealing session is a pure function of
        # its own key and cannot have moved since the seal was written.
        if entry.source in _SOURCES_WITHOUT_TIMESTAMPS:
            excluded["unstable_identity_source"] += 1
            continue
        if not entry.started_at or str(entry.started_at) > generated_at:
            excluded["not_provably_pre_sealing"] += 1
            continue
        if entry.group_size != 1 or entry.split_key != entry.session_key:
            excluded["grouped_identity"] += 1
            continue
        if bucket_for(entry.session_key) not in buckets:
            excluded["bucket_mismatch"] += 1
            continue
        if entry.session_key in dev_keys:
            excluded["named_by_dev_artifacts"] += 1
            continue
        end = _entry_end(entry)
        if end is None:
            excluded["no_end_timestamp"] += 1
            continue
        if end >= SESSION_END_BEFORE:
            excluded["ended_after_cutoff"] += 1
            continue
        if entry.bytes > MAX_SESSION_BYTES:
            excluded["too_large"] += 1
            continue
        eligible.append(entry)
    eligible.sort(
        key=lambda e: hashlib.sha256(e.session_key.encode("utf-8")).hexdigest()
    )
    chosen = eligible[:SESSIONS_PER_SPLIT]
    # Process in index (session_key) order — the same order the write CLI's
    # --session mode uses, so budget arithmetic is identical between runs.
    chosen.sort(key=lambda e: e.session_key)
    notes = {
        "split": split,
        "sessions_in_split": len(pool),
        "eligible": len(eligible),
        "excluded": excluded,
        "selected": len(chosen),
    }
    return chosen, notes


def _runner_config() -> RunnerConfig:
    return RunnerConfig(
        dry_run=True,
        max_ops_per_session=MAX_OPS_PER_SESSION,
        max_total_ops=MAX_TOTAL_OPS,
        insights=ExtractionConfig(),
    )


def cmd_extract(args: argparse.Namespace) -> int:
    split = args.split
    state = Path(args.state_dir)
    require(state / "preregistration.json", "preregister")
    if split == "holdout":
        seal = require(state / "seal-check.json", "seal-check")
        if not (
            seal.get("seal_unmoved")
            or (seal.get("membership_proof") or {}).get("available")
        ):
            raise SystemExit("seal not proven; the sealed split stays closed")

    # The consumption gate, before any session is read — the stage's own rule.
    ledger = Ledger(DEFAULT_STATE_DIR.expanduser())
    gate = evaluate_gate(
        Path(DEFAULT_BASELINE),
        Path(DEFAULT_VERDICT),
        None,
        written_nodes_total=ledger.written_nodes_total(),
        bootstrap_allowance=DEFAULT_BOOTSTRAP_ALLOWANCE,
    )
    if not gate.authorized:
        log(f"GATE {gate.status}: {'; '.join(gate.reasons)}")
        return 3
    log(f"GATE PASS ({gate.mode})")

    entries, notes = _select_sessions(split, state)
    log(f"{split}: {notes['eligible']} eligible, extracting from {len(entries)}")
    records = []
    failures = 0
    for entry in entries:
        try:
            records.append(load_session(entry))
        except Exception as exc:  # noqa: BLE001 - a corrupt transcript is a count
            failures += 1
            log(f"! load failed {entry.session_key}: {exc}")
    notes["loaded"] = len(records)
    notes["failed_to_load"] = failures

    cassette = state / f"cassette-{split}.json"
    config = _runner_config()
    runner = ExtractionRunner(
        config=config,
        ledger=ledger,
        client=None,
        judge_factory=lambda: LockingCassetteJudge(
            cassette, inner=ClaudeCliJudge(model=EXTRACT_JUDGE_MODEL), record=True
        ),
    )
    started = time.monotonic()
    if EXTRACT_WORKERS > 1 and len(records) > 1:
        with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as pool:
            proposals = list(pool.map(runner.propose, records))
    else:
        proposals = [runner.propose(record) for record in records]
    decisions = runner.decide_and_execute(proposals)
    elapsed = round(time.monotonic() - started, 1)

    report = build_report(
        mode="dry-run",
        splits_read=[split],
        gate_decision=gate,
        proposals=proposals,
        decisions=decisions,
        config=config,
        ledger=ledger,
        extra={"selection": notes, "elapsed_seconds": elapsed},
    )
    (state / f"extract-{split}.json").write_text(
        default_redactor(canonical_json(report)) + "\n", encoding="utf-8"
    )

    # Full op dump (attributed payloads, private, never tracked): what the
    # seeding and audit phases consume. Iteration order matches
    # decide_and_execute exactly, so ops and decisions zip 1:1.
    dump: list[dict[str, Any]] = []
    cursor = 0
    for proposal in proposals:
        for op in proposal.ops + proposal.attests:
            attributed = enforce_attribution(op, proposal.record, task=config.task)
            decision = decisions[cursor]
            cursor += 1
            if attributed.kind == "attest":
                continue
            expected = op_fingerprint(attributed)
            if expected != decision.fingerprint:
                raise SystemExit(
                    f"op/decision misalignment at #{cursor}: {expected[:12]} != "
                    f"{decision.fingerprint[:12]}"
                )
            dump.append(
                {
                    "kind": attributed.kind,
                    "session_key": proposal.record.session_key,
                    "fingerprint": decision.fingerprint,
                    "status": decision.status,
                    "reason": decision.reason,
                    "payload": attributed.payload,
                    "provenance": {
                        "session_key": attributed.provenance.session_key,
                        "transcript_sha256": attributed.provenance.transcript_sha256,
                        "source_span": attributed.provenance.source_span,
                    },
                }
            )
    write_json(state / f"ops-{split}.json", {"batch": BATCH, "split": split, "ops": dump})
    accepted = sum(1 for item in dump if item["status"] == "would_write")
    log(
        f"{split}: {len(dump)} node-writing ops proposed, {accepted} would_write, "
        f"judge ${report['judge']['total_cost_usd']}, {elapsed}s"
    )
    return 0


# ---------------------------------------------------------------------------
# seed-and-measure + control: the counterfactual protocol
# ---------------------------------------------------------------------------


def _counterfactual(argv: list[str]) -> int:
    command = [sys.executable, str(REPO_ROOT / "scripts/counterfactual_consumption.py")]
    log("$ " + " ".join(command[1:] + argv))
    return subprocess.run(command + argv, cwd=REPO_ROOT, check=False).returncode


def cmd_seed_and_measure(args: argparse.Namespace) -> int:
    split = args.split
    state = Path(args.state_dir)
    require(state / "preregistration.json", "preregister")
    ops_doc = require(state / f"ops-{split}.json", f"extract --split {split}")
    snapshot = state / "snapshot.sqlite3"
    if not snapshot.is_file():
        raise SystemExit(f"no frozen snapshot at {snapshot}; run freeze first")

    accepted = [
        op
        for op in ops_doc["ops"]
        if op["status"] == "would_write" and op["kind"] in ("remember", "teach")
    ]
    if not accepted:
        raise SystemExit(f"no accepted ops for split {split}; nothing to measure")
    batch_id = f"{BATCH}-{split}"

    seeded_db = state / f"seeded-{split}.sqlite3"
    log(f"copying frozen snapshot -> {seeded_db}…")
    backup_database(snapshot, seeded_db)
    store = MemoryStore(MemoryConfig(db_path=seeded_db))
    seeded: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    try:
        decayed_before = int(
            store.connection.execute("SELECT count(*) FROM nodes WHERE decayed=1").fetchone()[0]
        )
        for op in accepted:
            payload = op["payload"]
            context = dict(payload.get("context") or {})
            context["field_batch"] = batch_id
            if op["kind"] == "remember":
                node = store.create_node(
                    level="trace",
                    content=str(payload["content"]),
                    context=context,
                    timestamp=SEED_CREATED_AT,
                )
                node_id = node.id
            else:
                try:
                    taught = memory_teach(
                        store,
                        str(payload["trace_id"]),
                        payload["correction"],
                        confidence=payload.get("confidence"),
                        context=context,
                    )
                except KeyError:
                    skipped.append(
                        {
                            "fingerprint": op["fingerprint"],
                            "reason": "teach original absent from snapshot",
                        }
                    )
                    continue
                node_id = taught.corrective_trace.id
            with store.connection:
                store.connection.execute(
                    "UPDATE nodes SET created_at = ?, updated_at = ? WHERE id = ?",
                    (SEED_CREATED_AT, SEED_CREATED_AT, node_id),
                )
            seeded.append(
                {"fingerprint": op["fingerprint"], "kind": op["kind"], "node_id": node_id}
            )
        decayed_after = int(
            store.connection.execute("SELECT count(*) FROM nodes WHERE decayed=1").fetchone()[0]
        )
    finally:
        store.close()

    write_json(
        state / f"seeded-{split}.json",
        {
            "batch": batch_id,
            "split": split,
            "accepted_ops": len(accepted),
            "seeded_nodes": len(seeded),
            "skipped": skipped,
            "created_at_stamp": SEED_CREATED_AT,
            "decay_collateral": decayed_after - decayed_before,
            "nodes": seeded,
        },
    )
    log(f"seeded {len(seeded)} node(s) into {seeded_db.name} (skipped {len(skipped)})")

    rule = (
        f"every trace the dry-run extractor accepted from {SESSIONS_PER_SPLIT} "
        f"sealed {split} sessions ended before 2026-08-15, seeded at "
        f"{SEED_CREATED_AT}, marked context field_batch={batch_id}"
    )
    out = state / f"counterfactual-extracted-{split}.json"
    code = _counterfactual(
        [
            "--snapshot",
            str(seeded_db),
            "--as-of",
            AS_OF,
            "--window",
            COHORT_WINDOW,
            "--replay-since",
            REPLAY_SINCE,
            "--cohort-rule",
            rule,
            "--cohort-kind",
            f"extracted_{split}",
            "--context-key",
            f"field_batch={batch_id}",
            "--reinsert-mode",
            "fresh",
            "--allow-partial-replay",
            "--out",
            str(out),
        ]
    )
    # Exit 1 only means selfcheck.passed=false, which an extracted cohort's
    # zero retrospective arm makes meaningless; the file is the result.
    return 0 if out.is_file() and code in (0, 1) else 2


def cmd_control(args: argparse.Namespace) -> int:
    state = Path(args.state_dir)
    snapshot = state / "snapshot.sqlite3"
    if not snapshot.is_file():
        raise SystemExit(f"no frozen snapshot at {snapshot}; run freeze first")
    out = state / "counterfactual-control.json"
    code = _counterfactual(
        [
            "--snapshot",
            str(snapshot),
            "--as-of",
            AS_OF,
            "--window",
            COHORT_WINDOW,
            "--replay-since",
            REPLAY_SINCE,
            "--cohort-rule",
            CONTROL_COHORT_RULE,
            "--cohort-kind",
            "organic_holdout",
            "--reinsert-mode",
            "faithful",
            "--fresh-arm",
            "--allow-partial-replay",
            "--out",
            str(out),
        ]
    )
    return code if code in (0, 1) else 2


# ---------------------------------------------------------------------------
# audit: adversarial anti-dump precision on a deterministic sample
# ---------------------------------------------------------------------------


def cmd_audit(args: argparse.Namespace) -> int:
    split = args.split
    state = Path(args.state_dir)
    require(state / "preregistration.json", "preregister")
    ops_doc = require(state / f"ops-{split}.json", f"extract --split {split}")
    accepted = sorted(
        (
            op
            for op in ops_doc["ops"]
            if op["status"] == "would_write" and op["kind"] in ("remember", "teach")
        ),
        key=lambda op: op["fingerprint"],
    )
    sample = accepted[:AUDIT_SAMPLE_PER_SPLIT]
    judge = ClaudeCliJudge(model=AUDIT_JUDGE_MODEL)
    results: list[dict[str, Any]] = []
    genuine = dump = errors = 0
    for position, op in enumerate(sample, start=1):
        payload = op["payload"]
        if op["kind"] == "remember":
            trace = str(payload.get("content") or "")
        else:
            correction = payload.get("correction")
            trace = correction if isinstance(correction, str) else canonical_json(
                correction, indent=None
            )
        try:
            answer = judge.judge(
                AUDIT_TASK,
                AUDIT_SCHEMA,
                {"kind": op["kind"], "trace": trace},
                timeout_s=AUDIT_TIMEOUT_S,
            )
            verdict = str(answer.get("verdict"))
            reason = str(answer.get("reason") or "")
        except JudgeError as exc:
            verdict, reason = "error", str(exc)[:300]
        if verdict == "genuine":
            genuine += 1
        elif verdict == "dump":
            dump += 1
        else:
            errors += 1
        results.append(
            {
                "fingerprint": op["fingerprint"],
                "kind": op["kind"],
                "verdict": verdict,
                "reason": reason,
            }
        )
        log(f"  audit {split} {position}/{len(sample)}: {verdict}")
    audited = genuine + dump
    precision = round(genuine / audited, 4) if audited else None
    inconclusive = (
        not sample or (errors / max(1, len(sample))) > AUDIT_MAX_ERROR_SHARE
    )
    doc = {
        "artifact": "anti-dump-audit",
        "batch": BATCH,
        "split": split,
        "generated_at": utc_now(),
        "judge_model": AUDIT_JUDGE_MODEL,
        "task_sha256": PREREGISTRATION["audit"]["task_sha256"],
        "accepted_ops": len(accepted),
        "sampled": len(sample),
        "audited": audited,
        "genuine": genuine,
        "dump": dump,
        "errors": errors,
        "precision": precision,
        "inconclusive": inconclusive,
        "judge_cost_usd": round(judge.total_cost_usd, 4)
        if hasattr(judge, "total_cost_usd")
        else None,
        "results": results,
    }
    write_json(state / f"audit-{split}.json", doc)
    log(
        f"audit {split}: precision={precision} ({genuine}/{audited}), "
        f"errors={errors}, inconclusive={inconclusive}"
    )
    return 0


# ---------------------------------------------------------------------------
# verdict + report + write
# ---------------------------------------------------------------------------


def _wilson(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.96
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = (
        z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    )
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


def _rates(report: dict[str, Any]) -> dict[str, Any]:
    check = report["selfcheck"]
    grounded = check["grounded"]
    return {
        "cohort_kind": check["cohort_kind"],
        "cohort_nodes": check["cohort_nodes"],
        "counterfactual_rate": check["counterfactual_rate"],
        "counterfactual_consumed_later": check["counterfactual_consumed_later"],
        "rate_ci95": _wilson(
            int(check["counterfactual_consumed_later"]), int(check["cohort_nodes"])
        ),
        "grounded_rate": grounded["counterfactual_rate"],
        "grounded_denominator": grounded["counterfactual_denominator"],
        "events_replayed": report["protocol"]["request_stats"]["events_replayed"],
    }


def cmd_verdict(args: argparse.Namespace) -> int:
    state = Path(args.state_dir)
    prereg = require(state / "preregistration.json", "preregister")
    extracted_path = state / "counterfactual-extracted-holdout.json"
    control_path = state / "counterfactual-control.json"
    extracted = require(extracted_path, "seed-and-measure --split holdout")
    control = require(control_path, "control")
    extracted_eval = require(
        state / "counterfactual-extracted-eval.json", "seed-and-measure --split eval"
    )
    audit_holdout = require(state / "audit-holdout.json", "audit --split holdout")
    audit_eval = require(state / "audit-eval.json", "audit --split eval")
    seeded_holdout = require(
        state / "seeded-holdout.json", "seed-and-measure --split holdout"
    )
    cohort_big_enough = int(seeded_holdout["seeded_nodes"]) >= MIN_EXTRACTED_COHORT

    ledger = Ledger(DEFAULT_STATE_DIR.expanduser())
    gate = evaluate_gate(
        Path(DEFAULT_BASELINE),
        extracted_path,
        control_path,
        written_nodes_total=ledger.written_nodes_total(),
        bootstrap_allowance=DEFAULT_BOOTSTRAP_ALLOWANCE,
    )

    holdout_rates = _rates(extracted)
    eval_rates = _rates(extracted_eval)
    control_rates = _rates(control)
    organic_rate = float(control_rates["counterfactual_rate"] or 0.0)
    extracted_rate = float(holdout_rates["counterfactual_rate"] or 0.0)

    precision_holdout = audit_holdout.get("precision")
    precision_eval = audit_eval.get("precision")
    audit_ok = (
        not audit_holdout.get("inconclusive")
        and precision_holdout is not None
        and float(precision_holdout) >= AUDIT_PRECISION_BAR
    )
    degradation = None
    if (
        precision_eval
        and precision_holdout is not None
        and not audit_eval.get("inconclusive")
    ):
        degradation = round(
            (float(precision_eval) - float(precision_holdout)) / float(precision_eval), 4
        )
    degradation_ok = degradation is not None and degradation <= DEGRADATION_BAR

    eval_rate = float(eval_rates["counterfactual_rate"] or 0.0)
    consumption_degradation = (
        round((eval_rate - extracted_rate) / eval_rate, 4) if eval_rate else None
    )

    overall = (
        "PASS"
        if (gate.authorized and audit_ok and degradation_ok and cohort_big_enough)
        else "FAIL"
    )
    if (
        gate.status == "INCONCLUSIVE"
        or audit_holdout.get("inconclusive")
        or not cohort_big_enough
    ):
        overall = "INCONCLUSIVE"

    gate_dict = gate.to_dict()
    # The privacy guard bounds every published string at 200 printable ASCII
    # chars; gate reasons are prose and may exceed it, so they are chunked.
    gate_dict["reasons"] = [
        reason[start : start + 200]
        for reason in gate_dict["reasons"]
        for start in range(0, len(reason), 200)
    ]

    doc = {
        "artifact": "field-verdict",
        "batch": BATCH,
        "generated_at": utc_now(),
        "preregistration_sha256": prereg["plan_sha256"],
        "gate": gate_dict,
        "four_numbers": {
            "1_relative": {
                "extracted_rate": extracted_rate,
                "organic_rate": organic_rate,
                "ratio_vs_organic": round(extracted_rate / organic_rate, 4)
                if organic_rate
                else None,
                "bar": "extracted >= 0.5 x organic",
                "required": round(0.5 * organic_rate, 4),
                "pass": bool(organic_rate and extracted_rate >= 0.5 * organic_rate),
            },
            "2_absolute": {
                "extracted_rate": extracted_rate,
                "bar": "absolute_floor_by_protocol[counterfactual_post_cutoff] = 0.05",
                "pass": extracted_rate >= 0.05,
                "legacy_scalar_note": [
                    "the goal text's 0.15 is the deprecated retrospective-protocol "
                    "floor;",
                    "against it the extracted rate "
                    f"{'meets' if extracted_rate >= 0.15 else 'does not meet'} 0.15,",
                    "but the baseline forbids reading the scalar and the organic "
                    f"control itself scores {organic_rate} under this protocol",
                ],
            },
            "3_audit_precision": {
                "holdout_precision": precision_holdout,
                "ci95": _wilson(
                    int(audit_holdout.get("genuine") or 0),
                    int(audit_holdout.get("audited") or 0),
                ),
                "bar": AUDIT_PRECISION_BAR,
                "pass": bool(audit_ok),
            },
            "4_generalization": {
                "eval_precision": precision_eval,
                "holdout_precision": precision_holdout,
                "relative_degradation": degradation,
                "bar": DEGRADATION_BAR,
                "pass": bool(degradation_ok),
                "consumption_degradation_diagnostic": consumption_degradation,
            },
        },
        "cohort_size_rule": {
            "seeded_holdout_nodes": int(seeded_holdout["seeded_nodes"]),
            "minimum": MIN_EXTRACTED_COHORT,
            "pass": cohort_big_enough,
        },
        "grounded_variant": {
            "extracted": {
                "rate": holdout_rates["grounded_rate"],
                "denominator": holdout_rates["grounded_denominator"],
            },
            "organic": {
                "rate": control_rates["grounded_rate"],
                "denominator": control_rates["grounded_denominator"],
            },
            "note": "rate over nodes_with_closed_consumer per the baseline; "
            "compared rate to rate, never counts",
        },
        "cohorts": {
            "extracted_holdout": holdout_rates,
            "extracted_eval": eval_rates,
            "organic_control": control_rates,
        },
        "verdict": overall,
    }
    write_json(state / "verdict.json", doc)
    log(
        f"VERDICT {overall}: gate={gate.status} "
        f"extracted={extracted_rate} organic={organic_rate} "
        f"precision={precision_holdout} degradation={degradation}"
    )
    return 0 if overall == "PASS" else 1


def cmd_write(args: argparse.Namespace) -> int:
    """On PASS only: the bounded live write through the real runner CLI."""

    state = Path(args.state_dir)
    verdict = require(state / "verdict.json", "verdict")
    if verdict["verdict"] != "PASS":
        raise SystemExit(f"verdict is {verdict['verdict']}; zero live writes")
    ops_doc = require(state / "ops-holdout.json", "extract --split holdout")
    sessions = sorted({op["session_key"] for op in ops_doc["ops"]})
    argv = [
        sys.executable,
        str(REPO_ROOT / "scripts/post_session_extract.py"),
        "--i-am-the-sealed-measurement-run",
        "--write",
        "--judge",
        "cassette",
        "--cassette",
        str(state / "cassette-holdout.json"),
        "--max-total-ops",
        str(LIVE_WRITE_CAP),
        "--gate-verdict",
        str(state / "counterfactual-extracted-holdout.json"),
        "--gate-control",
        str(state / "counterfactual-control.json"),
        "--out",
        str(state / "holdout-write.json"),
        "--workers",
        "1",
    ]
    for key in sessions:
        argv += ["--session", key]
    log("$ " + " ".join(argv[1:8]) + f" … ({len(sessions)} sessions)")
    return subprocess.run(argv, cwd=REPO_ROOT, check=False).returncode


def _chunked(value: Any) -> Any:
    """Split >200-char strings into 200-char parts (the privacy guard's bound).

    Presentational only: the canonical pre-registration lives in this script's
    constants and is pinned by plan_sha256, which is computed over the
    unchunked canonical form.
    """

    if isinstance(value, str):
        if len(value) <= 200:
            return value
        return [value[start : start + 200] for start in range(0, len(value), 200)]
    if isinstance(value, list):
        return [_chunked(item) for item in value]
    if isinstance(value, dict):
        return {key: _chunked(item) for key, item in value.items()}
    return value


def _what_would_have_to_change(report: dict[str, Any]) -> list[list[str]]:
    """On a non-PASS verdict: the concrete deltas, computed from the numbers.

    Reporting prose only. Every bar, floor and selection rule is the
    pre-registered one; nothing here feeds back into any measurement phase.
    Each item is a list of <=200-char parts (the privacy guard's string bound);
    the markdown rendering joins the parts with spaces.
    """

    verdict = report["verdict"]
    numbers = verdict["four_numbers"]
    cohort = verdict["cohort_size_rule"]
    items: list[list[str]] = []
    if not cohort["pass"]:
        selection = report["extraction"]["holdout"]["selection"]
        seeded = int(cohort["seeded_holdout_nodes"])
        selected = max(1, int(selection["selected"]))
        eligible = int(selection["eligible"])
        needed = -(-int(cohort["minimum"]) * selected // max(1, seeded))
        items.append(
            [
                f"cohort: {seeded} seeded holdout traces < minimum "
                f"{cohort['minimum']}. At the observed yield ({seeded} accepted "
                f"ops per {selected} sessions) a decision-grade cohort needs "
                f"~{needed} eligible sessions;",
                f"the whole sealed holdout holds {eligible} today. The corpus must "
                "accumulate more finished sessions; lowering the minimum is only "
                "admissible in a fresh pre-registration for a new batch.",
            ]
        )
    relative = numbers["1_relative"]
    if not relative["pass"]:
        items.append(
            [
                f"consumption (relative): extracted {relative['extracted_rate']} < "
                f"required {relative['required']} (0.5 x organic "
                f"{relative['organic_rate']}). The extracted traces must become "
                "worth recalling, not the bar lower.",
            ]
        )
    absolute = numbers["2_absolute"]
    if not absolute["pass"]:
        items.append(
            [
                f"consumption (absolute): extracted {absolute['extracted_rate']} < "
                "the protocol floor 0.05.",
            ]
        )
    audit = numbers["3_audit_precision"]
    if not audit["pass"]:
        holdout_audit = report["audit"]["holdout"]
        audited = int(holdout_audit.get("audited") or 0)
        genuine = int(holdout_audit.get("genuine") or 0)
        needed_genuine = -(-audited * 70 // 100)
        items.append(
            [
                f"audit: holdout anti-dump precision {audit['holdout_precision']} "
                f"({genuine}/{audited}) is below {audit['bar']}; "
                f"{needed_genuine}/{audited} was required. Raising it means "
                "tightening the extractor's gate,",
                "which is extractor iteration: allowed against train/eval only, "
                "then re-measured on a NEW sealed batch - this batch's holdout "
                "sessions and numbers are burned.",
            ]
        )
    generalization = numbers["4_generalization"]
    if not generalization["pass"]:
        items.append(
            [
                f"generalization: eval->holdout precision degradation "
                f"{generalization['relative_degradation']} exceeds "
                f"{generalization['bar']}: the extractor fits the sessions it was "
                "developed against.",
            ]
        )
    items.append(
        [
            "invariant: no bar, floor, replay set or audit rule may be tuned to "
            "these numbers; any change re-runs the full protocol on a fresh "
            "sealed batch under a new pre-registration.",
        ]
    )
    return items


def _extract_summary(state: Path, split: str) -> dict[str, Any]:
    report = read_json(state / f"extract-{split}.json")
    return {
        "selection": report.get("selection"),
        "ops": report.get("ops"),
        "attestations_proposed": (report.get("attestations") or {}).get("proposed"),
        "judge": report.get("judge"),
        "elapsed_seconds": report.get("elapsed_seconds"),
        "gate_at_extraction": {
            "status": (report.get("gate") or {}).get("status"),
            "mode": (report.get("gate") or {}).get("mode"),
        },
    }


def cmd_report(args: argparse.Namespace) -> int:
    state = Path(args.state_dir)
    prereg = require(state / "preregistration.json", "preregister")
    seal = require(state / "seal-check.json", "seal-check")
    snapshot = require(state / "snapshot-manifest.json", "freeze")
    verdict = require(state / "verdict.json", "verdict")

    live_writes: dict[str, Any] = {"performed": False, "written_node_ids": [], "cap": LIVE_WRITE_CAP}
    write_report_path = state / "holdout-write.json"
    if write_report_path.is_file():
        write_report = read_json(write_report_path)
        node_ids = list(write_report.get("written_nodes") or [])
        live_writes = {
            "performed": True,
            "cap": LIVE_WRITE_CAP,
            "written_node_ids": node_ids,
            "written": len(node_ids),
            "teach_edges": write_report.get("teach_edges") or [],
            "ops_status": write_report.get("ops"),
            "reversal": [
                f'memory_forget id={node_id} reason="reverse {BATCH} field-measurement write"'
                for node_id in node_ids
            ],
        }
    elif verdict["verdict"] != "PASS":
        live_writes["reason"] = (
            f"verdict {verdict['verdict']}: the pre-registered bar was not cleared; "
            "the extractor signals and stops instead of writing"
        )

    inputs = {}
    for name in (
        "counterfactual-extracted-holdout.json",
        "counterfactual-extracted-eval.json",
        "counterfactual-control.json",
        "audit-holdout.json",
        "audit-eval.json",
        "seeded-holdout.json",
        "seeded-eval.json",
    ):
        path = state / name
        if path.is_file():
            inputs[name] = sha256_file(path)

    extraction = {
        split: _extract_summary(state, split) for split in ("eval", "holdout")
    }
    audits = {}
    for split in ("eval", "holdout"):
        audit = read_json(state / f"audit-{split}.json")
        audits[split] = {
            key: audit.get(key)
            for key in (
                "accepted_ops",
                "sampled",
                "audited",
                "genuine",
                "dump",
                "errors",
                "precision",
                "inconclusive",
                "judge_model",
            )
        }

    report = {
        "artifact": "post-session-extraction-field-report",
        "batch": BATCH,
        "generated_at": utc_now(),
        "question": [
            "do traces extracted from sealed holdout sessions get consumed by "
            "recorded later recall traffic",
            "at least half as often as organic memory, are they genuine one-fact "
            "insights,",
            "and does extraction quality survive eval -> holdout",
        ],
        "preregistration": _chunked(prereg),
        "seal": {
            "sealed_holdout_sha256": seal["sealed_holdout_sha256"],
            "digest_rederivable": seal["digest_rederivable"],
            "why_not_rederivable": seal["why_not_rederivable"],
            "membership_proof": seal["membership_proof"],
            "per_source_reconciliation": seal["per_source_reconciliation"],
            "checked_at": seal["checked_at"],
        },
        "snapshot": {
            "sha256": snapshot["snapshot_sha256"],
            "bytes": snapshot["snapshot_bytes"],
            "row_counts": snapshot["row_counts"],
            "captured_at": snapshot["captured_at"],
            "as_of_cut": AS_OF,
            "frozen_in_seconds": snapshot.get("frozen_in_seconds"),
        },
        "extraction": extraction,
        "seeding": {
            split: {
                key: read_json(state / f"seeded-{split}.json").get(key)
                for key in (
                    "accepted_ops",
                    "seeded_nodes",
                    "created_at_stamp",
                    "decay_collateral",
                )
            }
            for split in ("eval", "holdout")
        },
        "audit": audits,
        "verdict": {
            key: verdict[key]
            for key in (
                "gate",
                "four_numbers",
                "grounded_variant",
                "cohorts",
                "verdict",
                "generated_at",
            )
        },
        "live_writes": live_writes,
        "state_inputs_sha256": inputs,
        "what_would_have_to_change": (
            _what_would_have_to_change(
                {"verdict": verdict, "extraction": extraction, "audit": audits}
            )
            if verdict["verdict"] != "PASS"
            else []
        ),
        "privacy": (
            "aggregates, digests and node ids only; dry-run reports, op dumps and "
            "cassettes stay in the untracked state directory"
        ),
    }
    check_privacy(report)
    write_json(FIELD_REPORT_JSON, report)
    FIELD_REPORT_MD.write_text(_render_md(report), encoding="utf-8")
    log(f"wrote {FIELD_REPORT_JSON} and {FIELD_REPORT_MD}")
    return 0


def _render_md(report: dict[str, Any]) -> str:
    verdict = report["verdict"]
    numbers = verdict["four_numbers"]
    seal = report["seal"]
    lines = [
        "# Post-session extraction — sealed-holdout field report",
        "",
        f"Batch `{report['batch']}`, generated {report['generated_at']}. "
        f"**Verdict: {verdict['verdict']}** "
        f"(gate {verdict['gate']['status']}, mode {verdict['gate']['mode']}).",
        "",
        "Measured on the already-accumulated transcripts before scaling, because "
        "\"written\" is not \"used\". Pre-registration "
        f"`{report['preregistration']['plan_sha256'][:16]}…` was frozen before any "
        "holdout number existed. The sealed holdout digest "
        f"`{seal['sealed_holdout_sha256'][:16]}…` "
        + (
            "re-derived byte-identically."
            if seal["digest_rederivable"]
            else "could not be re-derived (AE archival renames path-derived "
            "ae_node_result keys and the CLI's rolling window pruned the oldest "
            "claude sliver — see `field-report.json#seal`); every measured session "
            "instead carries an individual sealed-membership proof: stable UUID "
            "key, started before the sealing instant, singleton identity group, "
            "own-key bucket in the sealed holdout buckets, and named by no "
            "development artifact."
        ),
        "",
        "## The four pre-registered numbers",
        "",
        "| # | Metric | Value | Bar | Pass |",
        "|---|--------|-------|-----|------|",
        (
            f"| 1 | extracted vs organic consumption | "
            f"{numbers['1_relative']['extracted_rate']} vs "
            f"{numbers['1_relative']['organic_rate']} "
            f"(ratio {numbers['1_relative']['ratio_vs_organic']}) | >= 0.5x organic "
            f"({numbers['1_relative']['required']}) | "
            f"{numbers['1_relative']['pass']} |"
        ),
        (
            f"| 2 | absolute extracted rate | {numbers['2_absolute']['extracted_rate']} "
            f"| >= 0.05 (protocol floor) | {numbers['2_absolute']['pass']} |"
        ),
        (
            f"| 3 | anti-dump audit precision (holdout) | "
            f"{numbers['3_audit_precision']['holdout_precision']} "
            f"(95% CI {numbers['3_audit_precision']['ci95']}) | >= 0.70 | "
            f"{numbers['3_audit_precision']['pass']} |"
        ),
        (
            f"| 4 | eval->holdout precision degradation | "
            f"{numbers['4_generalization']['relative_degradation']} | <= 0.25 | "
            f"{numbers['4_generalization']['pass']} |"
        ),
        "",
        "Legacy scalar floor: " + " ".join(numbers["2_absolute"]["legacy_scalar_note"]),
        "",
        "## Cohorts under one protocol",
        "",
        "| Cohort | Nodes | Consumption | 95% CI | Grounded |",
        "|--------|------:|------------:|--------|---------:|",
    ]
    for key, label in (
        ("extracted_holdout", "extracted (holdout)"),
        ("extracted_eval", "extracted (eval)"),
        ("organic_control", "organic control"),
    ):
        row = verdict["cohorts"][key]
        lines.append(
            f"| {label} | {row['cohort_nodes']} | {row['counterfactual_rate']} | "
            f"{row['rate_ci95']} | {row['grounded_rate']} "
            f"(n={row['grounded_denominator']}) |"
        )
    lines += [
        "",
        f"Consumption degradation eval->holdout (diagnostic): "
        f"{numbers['4_generalization']['consumption_degradation_diagnostic']}.",
        "",
        "## Live writes",
        "",
    ]
    live = report["live_writes"]
    if live.get("performed"):
        lines.append(
            f"{live['written']} node(s) written through the runner (cap {live['cap']}). "
            "Reversal commands (exact):"
        )
        lines += ["", "```"] + list(live["reversal"]) + ["```"]
    else:
        lines.append(f"None. {live.get('reason', '')}".rstrip())
    if report.get("what_would_have_to_change"):
        lines += ["", "## What would have to change", ""]
        lines += [
            "- " + " ".join(parts) for parts in report["what_would_have_to_change"]
        ]
    lines += [
        "",
        "## Protocol",
        "",
        f"- Snapshot `{report['snapshot']['sha256'][:16]}…` "
        f"({report['snapshot']['row_counts']['nodes']} nodes, "
        f"{report['snapshot']['row_counts']['recall_events']} recall events), "
        f"captured {report['snapshot']['captured_at']}, cut at as-of {AS_OF}.",
        f"- Replay set: every recorded recall event in [{REPLAY_SINCE}, {AS_OF}] — "
        "the arm pinned by the baseline's replay_set_rule.",
        f"- Extracted traces seeded with created_at {SEED_CREATED_AT}; cohorts "
        "selected by context marker under the identical window, tool, as-of, "
        "containment and horizon.",
        "- Full pre-registration, per-phase digests and untracked-state hashes: "
        "see `field-report.json`.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--state-dir",
        default=str(REPO_ROOT / "tmp" / BATCH),
        help="untracked working state (default: %(default)s)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler, needs_split in (
        ("seal-check", cmd_seal_check, False),
        ("freeze", cmd_freeze, False),
        ("preregister", cmd_preregister, False),
        ("extract", cmd_extract, True),
        ("seed-and-measure", cmd_seed_and_measure, True),
        ("control", cmd_control, False),
        ("audit", cmd_audit, True),
        ("verdict", cmd_verdict, False),
        ("write", cmd_write, False),
        ("report", cmd_report, False),
    ):
        command = sub.add_parser(name)
        command.set_defaults(handler=handler)
        if needs_split:
            command.add_argument("--split", choices=("eval", "holdout"), required=True)
    args = parser.parse_args(argv)
    os.chdir(REPO_ROOT)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
