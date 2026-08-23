#!/usr/bin/env python3
"""Simulate the drain's near-duplicate collapse on the pre-hygiene backup.

Before an operator may set ``LM_DRAIN_NEAR_DUP_SUPERSEDES``, a read and
classified report has to exist of what the drain would actually collapse on a
real corpus. This script produces that report, and it produces it from the
REAL code path rather than from a reimplementation of it.

Fidelity: the pairs come from the shipped pass
----------------------------------------------
The pairs in the artifact are the pairs
``MemoryRecallService._supersede_drained_near_dups`` decided to write. The
script builds a real ``MemoryRecallService`` over a copy of the backup, asks
for the scope chunk index exactly the way recall asks for it
(``service._scope_chunk_index(scope, service._chunk_corpus_revision())``), and
calls the pass per scope with ``store.create_connection`` replaced by a
recorder. Every veto is therefore inherited rather than restated: the
provenance ``source_traces`` guard, the corrections guard, the
already-superseded guard on both sides, the level/scope match, the length
guard, the cosine threshold, and the "established node absorbs the arrival"
direction all execute as shipped. Restating them here would be the one way to
produce a report that is confidently wrong.

Nothing is written to any corpus. Three independent guarantees, all asserted:

* the live database is never opened at all -- the substrate is the backup file,
  and it is opened ``mode=ro`` exactly once, to ``VACUUM INTO`` a copy;
* ``store.create_connection`` is an instance-level recorder for the whole run,
  so the pass's only write call never reaches storage;
* a SQLite authorizer on the copy's connection DENIES insert/update/delete on
  ``connections``, so a write that somehow escaped the recorder would raise
  rather than land. ``COUNT(*) FROM connections`` is compared before and after
  and must be identical.

How candidates are offered, and why it matters
----------------------------------------------
In production the pass receives ``freshly_chunked[scope]`` -- the handful of
nodes this one recall just gave vectors to. "The backlog" of a corpus that was
never drained is the maximal-coverage reading of that: **every active trace of
the scope is offered as an arrival**. That is what this script does, and it is
not a neutral choice, so it is stated in the artifact:

* ``_best_pooled_match`` defers arrivals -- it prefers a bearer that was
  already settled in the corpus and consults the drain's own arrivals only when
  the settled part of the scope offers no match above the floor. With every
  active trace offered, the settled part is the non-trace and decayed nodes
  only, so in practice the nominee comes from the deferred pass, i.e. from the
  plain arg-max over the scope. The one divergence from a production-sized
  arrival set: when some concept/schema node scores above the floor against a
  trace, this run nominates that concept (and then drops the pair on the
  ``bearer.level != node.level`` veto), where a small-arrival run could have
  nominated a settled trace instead. That direction loses pairs, never invents
  them, and the count of nominations lost that way is reported.
* candidates are sorted greatest-id-first inside the pass, so for a pair of
  traces the LATER ULID is offered first and is the one demoted: the older node
  bears. With everything offered at once, "established absorbs the arrival"
  therefore resolves to "the earlier-created node absorbs the later one".

Arms
----
Four runs of the same pass over the same copy, differing only in env:

===================  =========================  ============================
arm                  ``LM_DRAIN_NEAR_DUP_COSINE`` ``LM_NEAR_DUP_IDENTIFIER_VETO``
===================  =========================  ============================
``t0.99-veto-off``   0.99 (the shipped default) 0
``t0.99-veto-on``    0.99                       unset (default on)
``t0.95-veto-off``   0.95                       0
``t0.95-veto-on``    0.95                       unset
===================  =========================  ============================

The veto-OFF arms are the enumeration: everything the drain's semantics would
collapse if the identifier veto did not exist. The veto-ON arms are what the
shipped default would actually write. The band verdict is read off the pair:
``ge_0_99`` is automatable only when no pair that SURVIVES the veto at
threshold 0.99 is verdicted ``different_facts``.

The 0.95 arms are context only. They exist so a reader can see what the 0.99
floor buys; nothing in the acceptance depends on them.

Verdicts
--------
``VERDICTS`` below maps ``bearer_id->candidate_id`` to
(``verbatim_repeat``|``different_facts``, note). Each entry was assigned by
READING the two texts. The mechanical identifier diff is evidence for a
verdict, never a substitute for one: a pair with an empty identifier diff can
still be two different facts, and where that is true the note says so. The
script REFUSES to write the artifact if any pair lacks a verdict, so the
artifact can never describe a pair nobody read. ``--allow-missing-verdicts``
exists for the reading pass itself and marks the JSON as incomplete.

Usage::

    python3 scripts/drain_near_dup_simulate.py \\
        --json artifacts/near-dup/drain-simulation.json \\
        --md artifacts/near-dup/drain-simulation.md

This script does NOT enable the drain. ``LM_DRAIN_NEAR_DUP_SUPERSEDES`` is
never set here (the run asserts it is off in its own process), and the pass is
called directly rather than through the gate. The env line an operator would
use is documented in the artifact and nowhere executed.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    # The pairs must come from THIS checkout's pass, not from whatever else is
    # installed on the box.
    sys.path.insert(0, str(REPO_ROOT / "src"))

from living_memory.near_dup import (  # noqa: E402
    IDENTIFIER_VETO_ENV,
    extract_identifiers,
    identifiers_absent_from,
)
from living_memory.retrieval import (  # noqa: E402
    DEFAULT_DRAIN_NEAR_DUP_COSINE,
    DRAIN_NEAR_DUP_COSINE_ENV,
    DRAIN_NEAR_DUP_ENV,
    DRAIN_NEAR_DUP_KIND,
    MemoryRecallService,
    _pooled_scope_vectors,
    drain_near_dup_supersedes_enabled,
)
from living_memory.storage import MemoryStore  # noqa: E402

try:
    import numpy as _np
except ImportError:  # pragma: no cover - numpy is a normal runtime dep
    _np = None  # type: ignore[assignment]


PROGRAM = "drain_near_dup_simulate.py"

DEFAULT_BACKUP = (
    Path.home()
    / ".local"
    / "share"
    / "living-memory"
    / "global.pre-neardup-hygiene-20260823.sqlite3"
)
#: Never opened by this script, and named only so the guard below can refuse it.
LIVE_DB_NAMES = ("global.sqlite3", "global-alt.sqlite3", "alt.sqlite3")
DEFAULT_WORKDIR = Path("/tmp/lm-drainsim")

VERDICT_VERBATIM = "verbatim_repeat"
VERDICT_DIFFERENT = "different_facts"
VERDICTS_ALLOWED = (VERDICT_VERBATIM, VERDICT_DIFFERENT)

BAND_HIGH = "ge_0_99"
BAND_CONTEXT = "band_0_95_to_0_99"

ARM_HIGH_VETO_OFF = "t0.99-veto-off"
ARM_HIGH_VETO_ON = "t0.99-veto-on"
ARM_CONTEXT_VETO_OFF = "t0.95-veto-off"
ARM_CONTEXT_VETO_ON = "t0.95-veto-on"

#: ``(threshold, identifier_veto)`` per arm, in run order.
ARMS: tuple[tuple[str, float, bool], ...] = (
    (ARM_HIGH_VETO_OFF, 0.99, False),
    (ARM_HIGH_VETO_ON, 0.99, True),
    (ARM_CONTEXT_VETO_OFF, 0.95, False),
    (ARM_CONTEXT_VETO_ON, 0.95, True),
)

DEFAULT_PREVIEW_CHARS = 1600

#: The exact line an operator would add to enable the drain. Documented, never
#: executed: nothing in this repo sets it, which ``grep_drain_flag`` proves.
OPERATOR_ENV_LINE = "LM_DRAIN_NEAR_DUP_SUPERSEDES=1"
OPERATOR_ENV_BLOCK = (
    "LM_DRAIN_NEAR_DUP_SUPERSEDES=1      # the gate; unset/0 = drain writes chunks only\n"
    "LM_DRAIN_NEAR_DUP_COSINE=0.99       # the band this artifact classified (shipped default)\n"
    "LM_NEAR_DUP_IDENTIFIER_VETO=1       # the veto; default-on, do NOT set to 0 with the gate on\n"
)


class SimulationError(RuntimeError):
    """A refusal or a broken precondition; always fatal, always explained."""


# --------------------------------------------------------------------------
# Verdicts, assigned by reading both texts
# --------------------------------------------------------------------------

_D = VERDICT_DIFFERENT
_V = VERDICT_VERBATIM

#: ``"bearer_id->candidate_id" -> (verdict, note)``.
#:
#: Assigned by reading the two texts of each pair, not by reading its
#: identifier diff. Where the diff is empty and the pair is still two different
#: facts, or the diff is non-empty and the pair is still one fact said twice,
#: the note says which and why -- those are the entries that prove the verdict
#: is a reading and not a restatement of the mechanical signal.
#:
#: This ledger was emptied and rebuilt from scratch for the re-run under the
#: TOKEN-EQUALITY rule. Blocking a collapse leaves the candidate a root, which
#: changes bearer resolution downstream, so the pair SET is not stable across
#: rules and a verdict carried over by pair id would be a verdict about a pair
#: that may no longer mean the same thing. Nothing below was inherited.
VERDICTS: dict[str, tuple[str, str]] = {
    # ---------------- band >= 0.99 ----------------
    "01KSAZD1N0HKFG9TE4YQ78CAYJ->01KSAZD11GAJ1PX2C9AVPGQ1GA": (
        _D,
        "Two critique nodes carrying the same prompt and OPPOSITE outcomes: "
        "`completed (PASS)` over 525.1s against `completed (FAIL)` over "
        "632.4s. The veto blocks, but on the duration token `632.4s` -- "
        "PASS/FAIL is an all-caps word and no identifier rule sees it. Equal "
        "durations would have let this through; see the residual-class "
        "section.",
    ),
    "01KS81EPTT5AVNHRMA933ENKBC->01KSR8A5P3D6C2FEBME1MHXWT5": (
        _D,
        "Two different tree nodes: `.../tooltip-implementation/_verify` bears, "
        "`.../tooltip-experiment-implementation/_verify` is the candidate. "
        "Neither name is a token of the other, so the veto blocks.",
    ),
    "01KRYA6QAC7KAA3SP1N1JZJJMT->01KS18Y9GFY5N4ZPYDHJCZ71PR": (
        _V,
        "The same rejected alternative recorded twice, reworded: 'scoped to "
        "critique only and not perform sibling tasks' against 'and must not "
        "execute sibling tasks'. Neither text names its node, and no fact "
        "differs. Collapses, correctly -- one of the two honest repeats in "
        "this band.",
    ),
    "01KTRJCB5FQ7ZT2WKD02TQ20B3->01KTRR6WHFXFW16M691Q1N98E1": (
        _D,
        "`checkpoint-selected-profile-v2` bears, "
        "`checkpoint-selected-profile-v2-repaired` is the candidate: two tree "
        "nodes, each with its own recorded `OUTCOME fail`. This is the pair "
        "whose MIRROR escaped the substring rule at 0.99350; under token "
        "equality the mirror also vetoes, so the mirror is no longer produced "
        "at all and this direction blocks.",
    ),
    "01KRVQNC96TW2K70A3JENYZXMP->01KRX759ASWP7GKSG4NX3K0284": (
        _D,
        "Stagnation reported on two different goals: "
        "`.../octopus-capability-stack` against "
        "`.../octopus-capability-stack-v2`. Suffixed sibling names; token "
        "equality blocks.",
    ),
    "01KSB3T1DJRNS2SMCR9189FBZS->01KSB3ZT0S8E97Q53TH3TT6DR6": (
        _V,
        "'Latency benchmark probe at 2026-05-24 -- synthetic timing trace; "
        "safe to decay.' against the same line with `#2` after 'probe'. Both "
        "self-declare as synthetic and disposable and carry the same date and "
        "the same body. The `#2` is invisible to the veto (`#` is outside the "
        "token class, so the token is the one character `2`) -- harmless HERE "
        "because the pair is one fact, and named in the residual-class "
        "section because it is not harmless in general. Collapses, correctly "
        "-- the second honest repeat in this band.",
    ),
    "01KVRKQM12J1E3QZB1DM7550TX->01KVRQ3FPY8R649R1VG984KA5K": (
        _D,
        "`stock-contract-preservation-audit-reintegrate` bears, "
        "`stock-contract-preservation-audit` is the candidate: two tree nodes "
        "under the same parent, each with its own `OUTCOME pass`. This is the "
        "second pair that escaped the substring rule (the shorter path sat "
        "inside the longer bearer's text); under token equality it blocks.",
    ),
    "01KVRN4A6AFYBF3GQQP0TNY3KT->01KVRQAP2ADN6YWCRVKKYMMDC3": (
        _D,
        "`source-throughput-projection-reintegrate` against "
        "`source-throughput-projection-benchmark`: two sibling tree nodes, "
        "two recorded passes. Blocks.",
    ),
    # ---------------- band 0.95 - 0.99 (context only) ----------------
    "01KRRPZ1Y81TDQBB2B5M5FM2YH->01KRT6P4EPHQS2DD2SWQXPZT0A": (
        _D,
        "Two `[project-map]` snapshots of the same repo taken at different "
        "times: one has the `make dashboard-e2e` target and 48 root files, "
        "the other lacks it and has 50. Two states, not one.",
    ),
    "01KSQRZ18QF1AY3TE4FKSGZJRZ->01KSQK8ECMZPQQS28HWR3P14BT": (
        _D,
        "`.../tooltip-experiment-implementation/_critique` against the "
        "parent's `EZ-13771-tooltip-experiment/_critique`. Different nodes; "
        "the boilerplate critique prompt is what drives the cosine.",
    ),
    "01KRS0ZZ35Z11X5EWZ2KVE0GZR->01KRS10087DBPPZ6DG0GT0E60E": (
        _D,
        "`[file-chunk]` 4/6 (lines 378-498) against 6/6 (lines 615-705) of "
        "the same fixture: different content of the same file.",
    ),
    "01KRXG9E6GYN291EQJGWSB2RNF->01KRXJKFDR6H2JSYR2NBZEMNR7": (
        _D,
        "Stagnation on `goal-emit-hook-canonical-and-dashboard-toggle` "
        "against `no-infeasible-finish-the-goal`: two goals.",
    ),
    "01KS80ZEFV23M0RCGCH8ZHSHPC->01KS7X135RRBYH3HZJG7C23CV9": (
        _D,
        "`OUTCOME pass` against `OUTCOME fail` on the SAME node "
        "`playwright-figma-screenshots`; the 399-character texts are "
        "otherwise byte-identical. The veto has nothing to see and passes: "
        "this pair SURVIVES at 0.95. Residual class: outcome polarity.",
    ),
    "01KS93EKA45V7BKP7JBB034K4B->01KS97R8GQXW3EC48PWETVKCNS": (
        _D,
        "Two completion records of `regenerate-gpu-baseline-walltime` with "
        "different commits (3204301f / 21576439) and different baseline ids.",
    ),
    "01KT9SGNF0A9TS70NJK5K7WC3Z->01KT9ZPG3AT3KD63NTEVFT2BWC": (
        _V,
        "The same EZ-13771 reopen lesson said twice: same bug, same evidence "
        "file, same requirement, reworded. The veto BLOCKS -- on `value/fill`, "
        "a slash-joined prose phrase the extractor reads as a name. Measured "
        "price of the veto, not a save.",
    ),
    "01KS1NQ7GXFWAF5C8PWJS8EKG4->01KS1NQC1KS8SAHYBCTF0WXXRS": (
        _D,
        "Stagnation on `.../validate-review-inputs` against its child "
        "`.../validate-review-inputs/children/validate-blinded-public-"
        "restricted-split`. Parent and child are different goals; the child "
        "path is not a token of the parent's text.",
    ),
    "01KSQK8ECMZPQQS28HWR3P14BT->01KSRQ2YS0PMYKP3C1QBPNY7NE": (
        _D,
        "`EZ-13771-tooltip-experiment/_critique` against "
        "`.../final-verification/_critique`: two critique nodes.",
    ),
    "01KS0T2HK1FKZGH85X5QXWRYFV->01KRZ8EKDXN4H7JZCYFMFG8J6M": (
        _D,
        "Stagnation on `rise/children/efe-causality-and-anti-shortcut` "
        "against `breakthrough/children/external-review-and-claim-boundary`.",
    ),
    "01KRNWYMPWFHK9C2031EBJKW3H->01KRP2RBD1AQ6RC4QK7XASYBST": (
        _D,
        "`OUTCOME fail` against `OUTCOME pass` on `routesearch-redesign`; the "
        "403-character Russian body, Confluence link and Figma link are "
        "identical. Nothing for the veto to see -- SURVIVES at 0.95. "
        "Residual class: outcome polarity.",
    ),
    "01KSQ1DFNZ83GSGDKG4JVA4415->01KSR5FCJ9MK1AM94704ZJM7JX": (
        _D,
        "`.../scale-and-calibrate-gpu-profile/_verify` against the parent "
        "`.../scale-ocpa-capacity-profile-on-gpu/_verify`: two verify nodes.",
    ),
    "01KT9SGNF0A9TS70NJK5K7WC3Z->01KT9T6BBQYVZN5DW5XJYWD3C8": (
        _V,
        "The same reopen lesson again, 'acceptance must assert' reworded to "
        "'verification must assert'. Same bug, same evidence path. Both texts "
        "extract the same identifier set, so this honest repeat collapses.",
    ),
    "01KVRMA23KK80DMTVT11JWTQH3->01KVRPGQHBEBANSCG0F8R5RBP0": (
        _D,
        "`cuda-rank-scratch-reintegrate` against "
        "`cuda-rank-scratch-source-repair`: two sibling tree nodes.",
    ),
    "01KSQRZ18QF1AY3TE4FKSGZJRZ->01KSQWRG4ET22RZTC2TKSG0T11": (
        _D,
        "`.../tooltip-experiment-implementation/_critique` against its child "
        "`.../tooltip-experiment-view-ui-actions/_critique`.",
    ),
    "01KS7W7T241MMGFJPP9BVE1K74->01KS7WV2Q3XWTYQB1TMMY0KMXC": (
        _D,
        "`retrieval-feedback-stability/implement-feedback-fix/_verify` "
        "against the parent `retrieval-feedback-stability/_verify`.",
    ),
    "01KRVQNC96TW2K70A3JENYZXMP->01KRXJWQV9BJ56Z4GQSWB6B19H": (
        _D,
        "Stagnation on `octopus-capability-stack` against "
        "`octopus-toward-agi-pipeline`: two goals.",
    ),
    "01KRS0ZZ35Z11X5EWZ2KVE0GZR->01KRS0ZYEWWNVNN70MSRRCZWYC": (
        _D,
        "`[file-chunk]` 4/6 against 3/6 of the same fixture.",
    ),
    "01KV1JWWXN9PDVWK14JBVB3TET->01KV1KRH09QCRAQXWP6CVVHQT6": (
        _D,
        "`default-f32-benchmark-evidence` against `default-f32-profile-"
        "evidence`: two sibling nodes producing two different JSONs.",
    ),
    "01KRW09XRXC9Z15M826C9DZFR1->01KRW0A276M9TBN0SVYMD3S0ZQ": (
        _D,
        "Stagnation on `.../fix-lifecycle-fake-replay-pack-reference` against "
        "`.../rerun-capability-stack-verification`: two sibling goals.",
    ),
    "01KS9R2PGCX3PK1Z3Z34NSYG0E->01KS9R3QJA8JBJ7QFEBWDTABX8": (
        _D,
        "Two runs of the same cherry-pick task, both PASS, but 293.2s against "
        "589.7s: the tree summary's payload IS the measured duration, so two "
        "durations are two facts.",
    ),
    "01KS1AB6RPV3JBKADJFZFQ32ZW->01KS1BK2H7YT02PS8KTR6YZH2S": (
        _V,
        "One implementation of `build-parity-report-scorer`, same worktree, "
        "same date, same two files added, same thresholds -- written up "
        "twice. The veto BLOCKS on `execution/telemetry`, `same-envelope`, "
        "`Newcombe/Wilson`, `normalized-score`: slash- and hyphen-joined prose "
        "that names nothing. Measured price of the veto.",
    ),
    "01KS0T3R59675J3A8HXGAEZJV3->01KRW09SACKXETDR3XCDGB3WKX": (
        _D,
        "Stagnation-with-restructure on `rise/children/token-action-substrate` "
        "against `octopus-capability-stack/children/_verify`.",
    ),
    "01KRXG9E6GYN291EQJGWSB2RNF->01KRXJB5BAG6N6HKJDXVWN12G1": (
        _D,
        "Stagnation on `goal-emit-hook-canonical-and-dashboard-toggle` "
        "against `no-infeasible-policy-escalate-instead`.",
    ),
    "01KSQK8ECMZPQQS28HWR3P14BT->01KSRV8SQ69PGJ296XW4F16HMZ": (
        _D,
        "`EZ-13771-tooltip-experiment/_critique` against "
        "`.../playwright-review-recapture-repair/_critique`.",
    ),
    "01KS98AA0M2Q25C4102M4TE59V->01KS9AJF5EXN200JQC7MV8BASS": (
        _D,
        "`integrate-vectorized-efe-planner-into-live-verify` against "
        "`regenerate-gpu-preopt-baseline-walltime`: two nodes, two worktrees, "
        "two baseline artifacts.",
    ),
    "01KVGRFJCX6KATY5WRGHB8BBXZ->01KVGX12YTFDXFH0J6F7MVMY2Y": (
        _D,
        "`score-p15c-b06` against `score-p15c-b07`: different candidates, "
        "different ranks, different measured scores.",
    ),
    "01KT9VDSW0YEFZKQFJDNEQA1MJ->01KT9W4BK1JKTWK2KY6W6D8JZ8": (
        _V,
        "The same reopen lesson, longer rewording. Same bug, same evidence "
        "file. Identifier sets match, so it collapses -- an honest repeat.",
    ),
    "01KV4B73BJ7HBSCRM3QWHKDYTM->01KV5BP2ENKPSRK9GME40S2CMX": (
        _D,
        "Two `OUTCOME fail` records for the SAME node "
        "`expanded-curriculum-training-v2-fresh-evidence`, but the goal text "
        "was rewritten between attempts ('amended non-stale split protocol "
        "and the new raw-record exporter' -> 'refreshed protocol hashes, "
        "non-stale fresh split, and raw-record exporter'), so the two records "
        "describe two attempts under two contracts. Survives the veto; not in "
        "the >=0.99 band.",
    ),
    "01KT3THPSVF31R7MET6KJTVA8H->01KTV02NVWMZF49M3TF3XD104Z": (
        _D,
        "`OUTCOME fail` against `OUTCOME pass` on "
        "`EZ-13870-dashboard-redesign`; the 381-character bodies are "
        "identical. SURVIVES at 0.95. Residual class: outcome polarity.",
    ),
    "01KSYET73WPPFK5CYHXENBAENC->01KSYGQ9JB7BEPP8G00JRE2BGA": (
        _V,
        "Same node `sm-baseline-attribution`, same `OUTCOME fail`; the "
        "candidate's text only appends 'using realistic mock data'. One "
        "recorded outcome said twice.",
    ),
    "01KVRJKRB5XJJJD0J00RMT9EQR->01KVRP0W933R3C4ZBK4VYT0EVA": (
        _D,
        "`packet-bptt-accounting-guards` against "
        "`stock-packet-accounting-reload-guards`: two sibling nodes.",
    ),
    "01KT9W4BK1JKTWK2KY6W6D8JZ8->01KT9W9R2C1ZTH0TS2JJGAJDWH": (
        _V,
        "The same reopen lesson again. The veto BLOCKS on `coloring/fill` -- "
        "prose, not a name. Price of the veto.",
    ),
    "01KRWHQTBDBGQC6WJ1YGYTY6YT->01KRXZS1MJYZYVZ0H8JMRNG2QN": (
        _D,
        "Stagnation on `archive/node-skip-empty-diff-for-verify` against "
        "`cleanup-llm-wrapper-and-rewire-via-ocpa-planner`.",
    ),
    "01KRS0ZZ35Z11X5EWZ2KVE0GZR->01KRS0ZRZ4J2XDNW627QWEC7W4": (
        _D,
        "`[file-chunk]` of two DIFFERENT fixtures (`..._split_v1_...` against "
        "`..._quick_fixture_v1_...`), different sha256.",
    ),
    "01KT9ZPG3AT3KD63NTEVFT2BWC->01KTA24P3NVX47HXCS2269PQR1": (
        _V,
        "The same reopen lesson once more. Blocked on `filled-star`, a "
        "hyphenated adjective. Price of the veto.",
    ),
    "01KS95H5T1D5VD2GZ47GBFF49S->01KS97Q68YZKHZ3T1NY27XKBH1": (
        _D,
        "Two execution records of the same node at two different commits "
        "(5cd85ef5 / 7121af45) with two different preopt baseline ids.",
    ),
    "01KT9SGNF0A9TS70NJK5K7WC3Z->01KT9VDSW0YEFZKQFJDNEQA1MJ": (
        _V,
        "The same reopen lesson, third rewording. Identifier sets match; "
        "honest collapse.",
    ),
    "01KT9VDSW0YEFZKQFJDNEQA1MJ->01KT9WKGW680HRF1CY3NJ4WYWX": (
        _V,
        "The same reopen lesson, this time with the bug sentence in Russian. "
        "Same fact; honest collapse.",
    ),
    "01KS98AA0M2Q25C4102M4TE59V->01KS9A9MXW40DM7NSS43X0JZWN": (
        _D,
        "`integrate-vectorized-efe-planner-into-live-verify` against "
        "`integrate-vectorized-efe-planner-into-verify`: two nodes, two "
        "worktrees, two baselines.",
    ),
    "01KSRCNV043HHTY3JGVYMDW8TJ->01KSRE9QW0JZ1MVSZC72ACYB2H": (
        _D,
        "`.../p6-boundary-repair/_verify` against the parent "
        "`.../render-stability-repair/_verify`.",
    ),
    "01KS8PVNM3N2YA862ZXQ8G1QSB->01KS8PWBW0TQWT2BRG5FTQ86PH": (
        _D,
        "Two verify nodes distinguished only by their root: "
        "`programming-action-substrate` against `dialogue-action-substrate`, "
        "236.6s against 380.6s.",
    ),
    "01KRVAHTJAYHRSCBRWG23NCS7F->01KRVDYPX157SGC69PCDJ30QBW": (
        _D,
        "Stagnation on `system-integration-end-to-end` "
        "(restructures_count=43) against `prompt-engine-v2` "
        "(restructures_count=163).",
    ),
    "01KV8TBC07CHTMBANSRVARXDRF->01KV8R6F1PM30K4J45SA631CDY": (
        _D,
        "`v5-protocol-validator-reset` against `v4-protocol-validator-reset`: "
        "different namespaces, different bases, different frozen hashes.",
    ),
    "01KS98J9FFB1GNKC7R22DYKDS3->01KS9AYJP06KVE4X4BJ9N8RKBF": (
        _D,
        "Two verification turns of the same `_verify` node with different "
        "regenerated baselines (0f61643c / 43a902b5) and different measured "
        "wall times.",
    ),
    "01KS98J9FFB1GNKC7R22DYKDS3->01KS94MB2STV1GWA337CT7N56S": (
        _D,
        "Same shape: two verification turns, baselines 0f61643c against "
        "49a2a74d, wall times 26.555929 against 26.377399.",
    ),
    "01KS1191AMVCKVB3KHQW1QJAEE->01KS0QKC6GA350T4HKWM0R4BNS": (
        _D,
        "Two critique runs over the same boilerplate prompt with different "
        "findings (review-collection DAG against path-contract/order/"
        "anti-lookup) and 74.6s against 109.75s.",
    ),
    "01KRS1G6DGHCVTR1N2JCHCEDBS->01KRS1GA0APCM203G6AFGH37D8": (
        _D,
        "`[project-manifest]` chunk 2 against chunk 3: different files, "
        "different symbols.",
    ),
    "01KVDHWC4WCZ7M3YRED2FWHH9V->01KVDJ45N38GM2CQJ6QF0V6AZZ": (
        _D,
        "'Completed a08-runtime-checkpoint-smoke-v2' against 'Re-verified "
        "... after merge-gate failure feedback': two events on one node, the "
        "second reporting the gap the first left.",
    ),
    "01KRWHQKPCX9ZYHFZYVWCYJT5Z->01KRX4WCDG1YCWH2XF6SMGD7DX": (
        _D,
        "Stagnation on `projects/ae/.../archive/prompt-engine-v2` against "
        "`projects/lm/.../lm-server-restart-endpoint`: two projects.",
    ),
    "01KVRJHA2QYPX56APZMS9R21TR->01KVRMSQ6M920CAZGTW2MS1YWG": (
        _D,
        "`cuda-factorized-equivalence-guards` against the same name plus "
        "`-reintegrate`: two nodes. Suffix pair; token equality blocks.",
    ),
    "01KSMR8833BF6Q714W752YPHVA->01KSMSAPP3JJAV82729M17EV0V": (
        _D,
        "Parent verification in `..._verify-shared-decoder-parent` against "
        "tree-goal verification in `..._shared-ocpa-token-decoder__verify`: "
        "two worktrees, two check lists.",
    ),
    "01KS98AA0M2Q25C4102M4TE59V->01KS962WHBDTE6X9QRZJ81P6D5": (
        _D,
        "`...-into-live-verify` against `...-into-verify` again, different "
        "commit (2ac26500 / 60610bcf) and baseline.",
    ),
    "01KVRN4A6AFYBF3GQQP0TNY3KT->01KVRJZ9TYQ4VBTAMN19FE6TFK": (
        _D,
        "`source-throughput-projection-reintegrate` against "
        "`source-throughput-projection`: two nodes.",
    ),
    "01KS5HMWJJ237D6FFCEPWVFJHB->01KS5JZ6VQNB1GDGPY8TR65F28": (
        _V,
        "One EZ-13771 requirements-matrix result written twice: same files, "
        "same experiment, same groups, same 64x64 crop rule, same rows, same "
        "Confluence pages. The veto BLOCKS -- the second write spells the "
        "Figma node `4584-13916` where the first says only 'Figma URL', and "
        "adds `non-empty`. A cautious block on a genuinely extra token, and "
        "still the price of the veto on an honest repeat.",
    ),
    "01KS5JZ6VQNB1GDGPY8TR65F28->01KS5HMWJJ237D6FFCEPWVFJHB": (
        _V,
        "The MIRROR of the pair above, produced only in `t0.95-veto-on`: with "
        "the richer text blocked as a candidate it stays a root and then "
        "bears the poorer one, which loses no identifier, so the veto passes. "
        "Same one fact; the collapse is honest and the direction is the "
        "better-detailed text absorbing the thinner one.",
    ),
    "01KS969ESDE1D0Q5HBHT0BDFM2->01KS9BE9XS1SBPYSQR8MFP8Q73": (
        _D,
        "Two tree summaries of the same cherry-pick prompt, 360.779s against "
        "502.205s, with different reported end states.",
    ),
    "01KRS1G6DGHCVTR1N2JCHCEDBS->01KRS1G0R59DE97ZDY4J1W72V3": (
        _D,
        "`[project-manifest]` chunk 2 against chunk 1.",
    ),
    "01KS9AYJP06KVE4X4BJ9N8RKBF->01KS9BEF51A6AGCM3QY6VTV5RA": (
        _D,
        "Two verification turns, baselines 43a902b5 against ab8f8827, wall "
        "times 26.348318 against 26.302047.",
    ),
    "01KVHDAAX6B1MHJNERES076V3Q->01KVHDM4T4GESY5YR5W7W8WXY1": (
        _D,
        "`score-expanded-v2-seed-2026062001` against `...-2026062002`: two "
        "seeds, two scored checkpoints, two verdicts.",
    ),
    "01KRS109KSGEDX2DCCC2AFGC3K->01KRS10A6BJPXWXXYDXJQQMKQ3": (
        _D,
        "`[file-chunk]` 1/6 (lines 1-61) against 2/6 (lines 62-140) of "
        "`envs/stress_tasks.py`.",
    ),
    "01KWWHB79A84X915J0YR7WNKEY->01KZ8S65TXD79TJE7XV056Y2EE": (
        _D,
        "A dated confirmation of the tier_policy gate trap on one named node "
        "(`start-dashboard-marker-clear`, commit a9b1c757) against the "
        "MIGRATED named lesson `project_tree_node_critical_tier_gate_trap`, "
        "which carries the prescriptive 'correct node behavior' and 'system "
        "fix options' sections and is a link target for other memories. "
        "Demoting the named lesson under a case report hides the name other "
        "traces resolve to.",
    ),
    "01KS98AA0M2Q25C4102M4TE59V->01KS95H5T1D5VD2GZ47GBFF49S": (
        _D,
        "Two execution records of the same node at commits 2ac26500 against "
        "5cd85ef5 with different baselines and wall times.",
    ),
    "01KTA3NCKN4GJYEME42FBZSPE3->01KTA5KTCP2B7Y103W9NP53H1D": (
        _D,
        "`EZ-13370-rarity-package/rarity-source-package/rarity-source-package` "
        "against its parent `EZ-13370-rarity-package/rarity-source-package`, "
        "with different Russian descriptions ('создать пакет @webmaps/"
        "rarity@47.0.1' against 'перенести исходники'). The parent path is a "
        "PREFIX of the child's, so substring containment passed this pair and "
        "token equality blocks it.",
    ),
    "01KS5EYX0ADB13Q5A1CKQGS9G6->01KS5FWP7E5EC0YCCFSDJMA3GF": (
        _V,
        "One requirements-matrix acceptance rerun written twice: same "
        "worktree, same two files, same rows, same CloudFront 403. Identifier "
        "sets match; honest collapse.",
    ),
    "01KVDKN3FQ50EFC30QSEMDV77N->01KVDSMECAM29A7PDX8GZ6H6J6": (
        _D,
        "`a08-runtime-checkpoint-smoke-merge-repair` against "
        "`p15c-a08-runtime-checkpoint-smoke`: two nodes in two worktrees.",
    ),
    "01KVRQ3FPY8R649R1VG984KA5K->01KVRHR63M477YQ24P3821H65T": (
        _V,
        "A pair the previous run never produced. It exists only in "
        "`t0.95-veto-on`: blocking "
        "`01KVRKQM…->01KVRQ3F…` leaves 01KVRQ3F a root, and it then bears "
        "01KVRHR6. Both texts are `OUTCOME pass` on the SAME node path "
        "`stock-contract-preservation-audit`, one phrased as the audit and "
        "one as its reintegration. One node passing, recorded twice.",
    ),
    "01KW6WE0WDRXF17N1R9WRNXGW8->01KW6X7SC2TGJDGDV4M8G61ANG": (
        _D,
        "An engineering completion record for NW-9 `build-plastic-core` "
        "(interface the siblings consume, test counts) against the node's own "
        "`OUTCOME pass` goal statement, which names the node path and "
        "`mechanism_objective_spec.{json,md}` the first never mentions.",
    ),
    "01KS0T38SEE6713ZWN0JA45C47->01KS17D4E1NVF6QMJ3JJZE6JEQ": (
        _D,
        "Same goal, DIFFERENT diagnosis and DIFFERENT action: "
        "`seconds_since_last_acceptance_pass` -> restructure against "
        "`seconds_since_last_commit` -> abandon. The veto sees it because the "
        "metric name is an underscore slug -- the action word alone "
        "(restructure/abandon) would have been invisible.",
    ),
    "01KT9S3S7TAXRED02Y40VGX2EV->01KT9WF6ST66BMN50R1JVES3K4": (
        _V,
        "The same reopen lesson, last rewording in this family. Honest "
        "collapse.",
    ),
    "01KX34DMJECNK45KC1F8XY2836->01KX374MMT5N096HC3CQAWCX02": (
        _D,
        "`OUTCOME fail` against `OUTCOME pass` on "
        "`EZ-14119-benzin-native/benzin-marker-color`; the 162-character "
        "bodies are identical. SURVIVES at 0.95. Residual class: outcome "
        "polarity.",
    ),
    "01KRS128TH842BV8BXA0ZGAA79->01KRS12HZEZ2B4V5FNYZDS7C65": (
        _D,
        "`[file-chunk]` 24/116 against 39/116 of `evals/lifecycle_core.py`.",
    ),
    "01KV2947ZM9EPRHTQZ7Z55DP48->01KV2BQR4GB87MF0G0YS4ZWDQ0": (
        _D,
        "`return-curve-evidence-v2` against `-v3`: two preregistered sweeps.",
    ),
    "01KRS1E3XX8H48ZJ9678EBR849->01KRS1EMMJEF0X3VXW1PETNWT1": (
        _D,
        "`[file-summary]` of `tests/test_runtime.py` against "
        "`tests/test_runtime_ext.py`: different files, different symbols.",
    ),
    "01KW3P2ACW4BEVS43FVN9ZF10T->01KW4287B548HPC6V21VHSP73W": (
        _D,
        "`rescope-envelope-and-strengthen-validators` (fail) against "
        "`...-validator` (pass): the names differ by one letter AND the "
        "outcome flips. The veto blocks on the name; the flip itself is "
        "invisible to it.",
    ),
    "01KTA10FPA55GF5Y1WBZD3NMCE->01KT9ZCGZ5KJ62RBEZ2592AZ8F": (
        _D,
        "Parent `gpu-backend` verified at HEAD 8b77b4d against child "
        "`cuda-backend-impl` complete at HEAD c4aeaf9: two nodes, two gate "
        "runs, different test counts.",
    ),
    "01KTBXTAP97M36K2MAKFW5DSK9->01KTBYKNVF4YWZED016PH7T1VT": (
        _D,
        "Tensor-core precision leaf complete (FAST_16F kept) against its "
        "merge-gate follow-up (backed off to FAST_TF32): opposite outcomes on "
        "the same file.",
    ),
    "01KVAFM7RY4GC779MTS70NR0WM->01KVAE3DPXV9NPCQTD5YRN0KT6": (
        _D,
        "`score-bsb-c04` against `score-bsb-c03`: different candidates, "
        "different opt_t, different battery results.",
    ),
    "01KSQG2M6PJ3KQSN67PMBSQ46D->01KSQ3BX7VD6FY7K55BYJGM7QQ": (
        _D,
        "A calibration-envelope diagnostic against the verification of the "
        "node above it: two records, two runs, two artifact sets.",
    ),
    "01KRYAJ4M7M3S8YXCGJNBA0CJM->01KRYC860HQVM92Y5KZ3RQ09FN": (
        _D,
        "`gemini-flash-contract`: 'Added contract.json and source_review.md' "
        "against 'Verified and tightened ... direct execution', with "
        "different acceptance evidence. Two events on one node.",
    ),
    "01KS18Y9GFY5N4ZPYDHJCZ71PR->01KS1KFEBXREX4FXPN4NR1ZYH8": (
        _V,
        "The rejected-alternative record again, 'implementation tasks' "
        "against 'implementation fixes', 'scoped to critique only' against "
        "'a critique-only child'. One fact. The veto BLOCKS on "
        "`critique-only` -- a hyphenated adjective. Price of the veto.",
    ),
    "01KT7H8JHXR5W2EZ3M6532EN03->01KT7TV1EC1VDB02PV5JJEAT4Y": (
        _D,
        "The original programming-BPE lifecycle wiring (2026-06-03) against "
        "its RE-CREATION from blueprint at commit f20d2b8f (2026-06-04), "
        "which explicitly cites the first by ULID as lost work.",
    ),
    "01KRS1E3XX8H48ZJ9678EBR849->01KRS1C3DYJP8YVZT3TF00SKMK": (
        _D,
        "`[file-summary]` of `tests/test_runtime.py` against "
        "`tests/test_evals_full_lifecycle_api.py`.",
    ),
    "01KWPN6CAQ3GEA6ZPDYF3EYPD4->01KZ8S2JP2PMKQ4JSWQFATR1EE": (
        _D,
        "The dated HTTPS-listener implementation note against the MIGRATED "
        "named lesson `project_dashboard_public_https`, which carries its own "
        "name, a `[[dashboard-split-test-migration]]` link and a distinct "
        "'pre-existing failures' framing. Demoting the named node hides the "
        "name other traces resolve to.",
    ),
    "01KTRJCB5FQ7ZT2WKD02TQ20B3->01KTRECZGVHN4T07WBB0TTXDC5": (
        _D,
        "`checkpoint-selected-profile-v2` against "
        "`checkpoint-selected-profile`: two nodes whose names are a prefix "
        "pair, with different goal texts. Substring containment passed this; "
        "token equality blocks it.",
    ),
    "01KVDNGK7YBTD8Q58EC84B9CFK->01KVDJJQ1WVDQYH36FRZ2BW2ND": (
        _D,
        "`a08-runtime-smoke-bounded-manifest` against "
        "`a08-runtime-checkpoint-smoke-v2`: two nodes.",
    ),
    "01KV8PQZQGJR91EBW2NXA7VBD7->01KV8VK3HBTS21NG9APW522GKW": (
        _D,
        "`fixed-epoch16-training-checkpoints` under one parent against "
        "`v5-fixed-epoch-scoreblind-training` under another: two nodes, two "
        "namespaces, two checkpoint hashes.",
    ),
    "01KSR5E34XKRN7NYWCHGF31SN7->01KSQZEPV2HNQ4KSWWT10VD053": (
        _D,
        "The `_verify` node's verification against "
        "`final-capacity-evidence`'s: two nodes, two HEADs, two run ids.",
    ),
}


# --------------------------------------------------------------------------
# Refusals: the live corpora are never touched
# --------------------------------------------------------------------------


def refuse_live_database(path: Path) -> None:
    """Raise if ``path`` names a live corpus or would be written beside one."""

    resolved = path.expanduser().resolve()
    if resolved.name in LIVE_DB_NAMES:
        raise SimulationError(f"refusing to open a live database: {resolved}")


def refuse_live_directory(path: Path, substrate: Path) -> None:
    """Raise if a read-write path sits in the live database's own directory.

    Guarding the directory rather than the file also protects the ``-wal`` and
    ``-shm`` beside it, and makes ``--workdir ~/.local/share/living-memory``
    impossible rather than merely unwise.
    """

    resolved = path.expanduser().resolve()
    live_dir = substrate.expanduser().resolve().parent
    if resolved == live_dir or live_dir in resolved.parents:
        raise SimulationError(
            f"refusing to write inside the live database's directory: {resolved}"
        )


def refuse_enabled_drain_flag() -> None:
    """Raise if this process has the drain gate on.

    The simulation calls the pass directly, so the gate is not what makes it
    run -- but a process that has the gate set is a process where an accidental
    ``memory_recall`` would mutate a corpus, and this script promises it is not
    that process.
    """

    if drain_near_dup_supersedes_enabled():
        raise SimulationError(
            f"{DRAIN_NEAR_DUP_ENV} is enabled in this process; this simulation "
            "refuses to run anywhere the drain could fire for real"
        )


# --------------------------------------------------------------------------
# The copy
# --------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def vacuum_copy(substrate: Path, destination: Path) -> dict[str, Any]:
    """``VACUUM INTO`` a ``mode=ro`` connection to the backup. Never writes it.

    ``cp`` is not equivalent: a database with a hot WAL copies as half a
    database, and ``VACUUM INTO`` folds the WAL in while reading through a
    connection that cannot write.
    """

    refuse_live_database(substrate)
    refuse_live_database(destination)
    refuse_live_directory(destination, substrate)
    if not substrate.exists():
        raise SimulationError(f"substrate not found: {substrate}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        stale = Path(str(destination) + suffix)
        if stale.exists():
            stale.unlink()
    started = time.perf_counter()
    connection = sqlite3.connect(f"file:{substrate}?mode=ro", uri=True)
    try:
        connection.execute("VACUUM INTO ?", (str(destination),))
    finally:
        connection.close()
    elapsed = time.perf_counter() - started
    return {
        "substrate": str(substrate),
        "substrate_opened": "read-only (file:...?mode=ro), VACUUM INTO a copy",
        "substrate_bytes": substrate.stat().st_size,
        "copy": str(destination),
        "copy_bytes": destination.stat().st_size,
        "copy_sha256": file_sha256(destination),
        "vacuum_seconds": round(elapsed, 3),
        **corpus_counts(destination),
    }


def corpus_counts(database: Path) -> dict[str, int]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        one = connection.execute
        return {
            "nodes": int(one("SELECT COUNT(*) FROM nodes").fetchone()[0]),
            "active_nodes": int(
                one("SELECT COUNT(*) FROM nodes WHERE decayed = 0").fetchone()[0]
            ),
            "active_traces": int(
                one(
                    "SELECT COUNT(*) FROM nodes WHERE decayed = 0 AND level = 'trace'"
                ).fetchone()[0]
            ),
            "chunk_rows": int(
                one("SELECT COUNT(*) FROM node_chunk_embeddings").fetchone()[0]
            ),
            "connections": int(one("SELECT COUNT(*) FROM connections").fetchone()[0]),
            "supersedes_edges": int(
                one(
                    "SELECT COUNT(*) FROM connections WHERE type = 'supersedes'"
                ).fetchone()[0]
            ),
        }
    finally:
        connection.close()


# --------------------------------------------------------------------------
# The write interception
# --------------------------------------------------------------------------


@dataclass(slots=True)
class RecordedEdge:
    """One ``create_connection`` call the pass made, recorded instead of written."""

    bearer_id: str
    candidate_id: str
    relation_type: str
    weight: float
    metadata: dict[str, Any]


class ConnectionRecorder:
    """Stands in for ``MemoryStore.create_connection`` for the whole run.

    Installed as an INSTANCE attribute, which shadows the bound method the pass
    reaches through ``self.store.create_connection``. The pass ignores the
    return value, so returning ``None`` changes nothing about its control flow;
    what it does after the call -- recording ``bearer_of`` and appending to its
    result list -- runs exactly as shipped, which is what keeps a third twin
    resolving onto the root rather than onto the copy that already lost.
    """

    def __init__(self) -> None:
        self.edges: list[RecordedEdge] = []

    def __call__(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        *,
        weight: float = 1.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.edges.append(
            RecordedEdge(
                bearer_id=str(source_id),
                candidate_id=str(target_id),
                relation_type=str(relation_type),
                weight=float(weight),
                metadata=dict(metadata or {}),
            )
        )
        return None


def install_connection_write_guard(store: MemoryStore) -> None:
    """DENY insert/update/delete on ``connections`` at the SQLite layer.

    Belt to the recorder's braces: if some path other than
    ``create_connection`` ever tried to write an edge, this raises instead of
    letting the row land. Installed AFTER ``MemoryStore.__init__`` so schema
    initialization and the retrieval-weight seed, which are writes to other
    tables on this scratch copy, are unaffected.
    """

    write_actions = (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE)

    def authorizer(
        action: int,
        arg1: str | None,
        arg2: str | None,
        db_name: str | None,
        source: str | None,
    ) -> int:
        if action in write_actions and arg1 == "connections":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    store.connection.set_authorizer(authorizer)


@contextmanager
def environment(values: Mapping[str, str | None]) -> Iterator[None]:
    """Set/unset env for one arm and restore it, whatever happens."""

    previous = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# --------------------------------------------------------------------------
# The simulation
# --------------------------------------------------------------------------


def scopes_with_active_traces(store: MemoryStore) -> list[str]:
    rows = store.connection.execute(
        """
        SELECT scope, COUNT(*) AS n FROM nodes
        WHERE decayed = 0 AND level = 'trace'
        GROUP BY scope ORDER BY n DESC, scope ASC
        """
    ).fetchall()
    return [str(row["scope"]) for row in rows]


def active_trace_nodes(store: MemoryStore, scope: str) -> list[Any]:
    """Every active trace of ``scope``, offered to the pass as an arrival.

    The maximal-coverage reading of "the backlog": a corpus that was never
    drained has every one of these still unsorted, and the pass's own filter
    (``node.level == 'trace' and not node.decayed``) is the same predicate, so
    nothing here narrows what the shipped code would consider.
    """

    rows = store.connection.execute(
        "SELECT id FROM nodes WHERE decayed = 0 AND level = 'trace' AND scope = ? "
        "ORDER BY id",
        (scope,),
    ).fetchall()
    node_ids = [str(row["id"]) for row in rows]
    fetched = store.get_nodes(node_ids)
    return [fetched[node_id] for node_id in node_ids if node_id in fetched]


@dataclass(slots=True)
class ArmResult:
    name: str
    threshold: float
    identifier_veto: bool
    env: dict[str, str | None]
    edges: list[RecordedEdge] = field(default_factory=list)
    seconds: float = 0.0
    scopes_run: int = 0
    candidates_offered: int = 0


def run_arm(
    store: MemoryStore,
    name: str,
    threshold: float,
    identifier_veto: bool,
    scopes: Sequence[str],
    *,
    progress: bool = True,
) -> ArmResult:
    """One full pass of the shipped drain semantics over every scope."""

    env: dict[str, str | None] = {
        DRAIN_NEAR_DUP_COSINE_ENV: f"{threshold}",
        IDENTIFIER_VETO_ENV: None if identifier_veto else "0",
    }
    recorder = ConnectionRecorder()
    store.create_connection = recorder  # type: ignore[method-assign]
    result = ArmResult(
        name=name, threshold=threshold, identifier_veto=identifier_veto, env=dict(env)
    )
    seen_edges = 0
    try:
        with environment(env):
            # A fresh service per arm: the chunk-index cache is per service and
            # the arms must not be able to inherit each other's state. The env
            # the pass reads is read inside the pass, per call, exactly as in
            # production.
            service = MemoryRecallService(store)
            started = time.perf_counter()
            for index, scope in enumerate(scopes, start=1):
                nodes = active_trace_nodes(store, scope)
                if not nodes:
                    continue
                # Verbatim the call the goal specifies, and verbatim what
                # ``_collect_vector`` does: revision first, index by that
                # revision, pass per scope.
                chunk_index = service._scope_chunk_index(
                    scope, service._chunk_corpus_revision()
                )
                written = service._supersede_drained_near_dups(scope, nodes, chunk_index)
                result.scopes_run += 1
                result.candidates_offered += len(nodes)
                # The pass's own return value and the recorder must agree, or
                # the interception is not seeing every write it makes.
                intercepted = len(recorder.edges) - seen_edges
                seen_edges = len(recorder.edges)
                if len(written) != intercepted:
                    raise SimulationError(
                        f"{name}/{scope}: pass reported {len(written)} pairs but "
                        f"{intercepted} were intercepted"
                    )
                if progress and (written or index % 10 == 0):
                    elapsed = time.perf_counter() - started
                    print(
                        f"  [{name}] {index}/{len(scopes)} {scope}: "
                        f"{len(nodes)} arrivals, {len(written)} pairs "
                        f"({elapsed:.0f}s)",
                        flush=True,
                    )
            result.seconds = time.perf_counter() - started
    finally:
        del store.create_connection  # restore the bound method
    result.edges = recorder.edges
    return result


# --------------------------------------------------------------------------
# Pair records
# --------------------------------------------------------------------------


def preview(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


@dataclass(slots=True)
class PairRecord:
    pair_id: str
    bearer_id: str
    candidate_id: str
    scope: str
    level: str
    cosine: float
    band: str
    bearer_chars: int
    candidate_chars: int
    bearer_text: str
    candidate_text: str
    bearer_text_truncated: bool
    candidate_text_truncated: bool
    identifiers_lost: tuple[str, ...]
    identifiers_bearer_only: tuple[str, ...]
    identifier_veto_blocks: bool
    arms: list[str]
    #: The untruncated texts. ``bearer_text``/``candidate_text`` above are
    #: previews and are what the artifact carries; every measurement --
    #: the identifier diff, the residual-class scan -- runs on these instead,
    #: because a difference past the preview cutoff is still a difference the
    #: drain would hide. Deliberately absent from :meth:`to_dict`: the JSON
    #: would double in size to repeat what the preview already shows.
    bearer_full_text: str = ""
    candidate_full_text: str = ""
    verdict: str | None = None
    verdict_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "bearer_id": self.bearer_id,
            "candidate_id": self.candidate_id,
            "scope": self.scope,
            "level": self.level,
            "cosine": self.cosine,
            "band": self.band,
            "bearer_chars": self.bearer_chars,
            "candidate_chars": self.candidate_chars,
            "bearer_text": self.bearer_text,
            "candidate_text": self.candidate_text,
            "bearer_text_truncated": self.bearer_text_truncated,
            "candidate_text_truncated": self.candidate_text_truncated,
            "identifiers_lost_by_collapse": list(self.identifiers_lost),
            "identifiers_bearer_only": list(self.identifiers_bearer_only),
            "identifier_veto_blocks": self.identifier_veto_blocks,
            "arms": sorted(self.arms),
            "survives_veto_at_0_99": ARM_HIGH_VETO_ON in self.arms,
            "verdict": self.verdict,
            "verdict_note": self.verdict_note,
        }


def band_of(cosine_value: float) -> str:
    return BAND_HIGH if cosine_value > 0.99 else BAND_CONTEXT


def build_pairs(
    store: MemoryStore,
    arms: Sequence[ArmResult],
    *,
    preview_chars: int,
) -> list[PairRecord]:
    """One record per distinct ``(bearer, candidate)`` pair over all arms."""

    by_pair: dict[str, PairRecord] = {}
    wanted: set[str] = set()
    for arm in arms:
        for edge in arm.edges:
            wanted.add(edge.bearer_id)
            wanted.add(edge.candidate_id)
    nodes = store.get_nodes(sorted(wanted))
    for arm in arms:
        for edge in arm.edges:
            pair_id = f"{edge.bearer_id}->{edge.candidate_id}"
            existing = by_pair.get(pair_id)
            if existing is not None:
                existing.arms.append(arm.name)
                continue
            bearer = nodes.get(edge.bearer_id)
            candidate = nodes.get(edge.candidate_id)
            if bearer is None or candidate is None:
                raise SimulationError(f"{pair_id}: a node of the pair is gone")
            bearer_text, bearer_cut = preview(bearer.content, preview_chars)
            candidate_text, candidate_cut = preview(candidate.content, preview_chars)
            # The veto's OWN extractor, on the full texts rather than the
            # previews: a report that measured the preview would understate the
            # diff exactly where the texts are long.
            lost = identifiers_absent_from(candidate.content, bearer.content)
            bearer_only = identifiers_absent_from(bearer.content, candidate.content)
            cosine_value = float(edge.metadata.get("cosine", 0.0))
            by_pair[pair_id] = PairRecord(
                pair_id=pair_id,
                bearer_id=edge.bearer_id,
                candidate_id=edge.candidate_id,
                scope=str(edge.metadata.get("scope", candidate.scope)),
                level=str(candidate.level),
                cosine=cosine_value,
                band=band_of(cosine_value),
                bearer_chars=len(bearer.content),
                candidate_chars=len(candidate.content),
                bearer_text=bearer_text,
                candidate_text=candidate_text,
                bearer_text_truncated=bearer_cut,
                candidate_text_truncated=candidate_cut,
                identifiers_lost=lost,
                identifiers_bearer_only=bearer_only,
                identifier_veto_blocks=bool(lost),
                arms=[arm.name],
                bearer_full_text=bearer.content,
                candidate_full_text=candidate.content,
            )
    ordered = sorted(by_pair.values(), key=lambda pair: (-pair.cosine, pair.pair_id))
    for pair in ordered:
        assigned = VERDICTS.get(pair.pair_id)
        if assigned is None:
            continue
        verdict, note = assigned
        if verdict not in VERDICTS_ALLOWED:
            raise SimulationError(f"{pair.pair_id}: unknown verdict {verdict!r}")
        pair.verdict = verdict
        pair.verdict_note = note
    return ordered


# --------------------------------------------------------------------------
# The band decision
# --------------------------------------------------------------------------


def band_summary(pairs: Sequence[PairRecord], band: str, survivor_arm: str) -> dict[str, Any]:
    in_band = [pair for pair in pairs if pair.band == band]
    survivors = [pair for pair in in_band if survivor_arm in pair.arms]
    different = [pair for pair in in_band if pair.verdict == VERDICT_DIFFERENT]
    surviving_different = [pair for pair in survivors if pair.verdict == VERDICT_DIFFERENT]
    blocked_different = [
        pair for pair in different if pair.identifier_veto_blocks
    ]
    verbatim_blocked = [
        pair
        for pair in in_band
        if pair.verdict == VERDICT_VERBATIM and pair.identifier_veto_blocks
    ]
    return {
        "band": band,
        "would_collapse_pairs_veto_off": len(in_band),
        "verdict_verbatim_repeat": len([p for p in in_band if p.verdict == VERDICT_VERBATIM]),
        "verdict_different_facts": len(different),
        "unverdicted": len([p for p in in_band if p.verdict is None]),
        "identifier_veto_blocks": len([p for p in in_band if p.identifier_veto_blocks]),
        "different_facts_blocked_by_veto": len(blocked_different),
        "different_facts_surviving_veto": len(surviving_different),
        "different_facts_surviving_ids": [pair.pair_id for pair in surviving_different],
        "verbatim_repeats_blocked_by_veto": len(verbatim_blocked),
        "surviving_pairs_after_veto": len(survivors),
        "automatable": bool(in_band) and not surviving_different
        and not [p for p in in_band if p.verdict is None],
    }


# --------------------------------------------------------------------------
# Residual classes the identifier veto structurally cannot see
# --------------------------------------------------------------------------
#
# "Zero different_facts pairs survived the veto" is a MEASUREMENT, and a
# measurement is not a guarantee. The veto only ever compares identifier
# tokens, so any difference carried by something that is not an identifier
# token is invisible to it BY CONSTRUCTION -- no threshold, no corpus and no
# tightening of the comparison changes that. The classes below are the ones
# this corpus actually contains; each is detected mechanically here, counted
# per band, and its invisibility is PROVEN against the shipped
# ``extract_identifiers`` rather than asserted in prose.


#: ``completed (PASS)`` / ``completed (FAIL)`` -- the tree-summary shape.
_POLARITY_COMPLETED_RE = re.compile(r"completed \((PASS|FAIL)\)")
#: ``OUTCOME pass:`` / ``OUTCOME fail:`` -- the node-outcome shape.
_POLARITY_OUTCOME_RE = re.compile(r"\bOUTCOME (pass|fail)\b")
#: ``#2``, ``#12`` -- an ordinal whose ``#`` is outside the token character
#: class, so the token the extractor sees is the bare digit run.
_ORDINAL_RE = re.compile(r"#(\d+)")

RESIDUAL_OUTCOME_POLARITY = "outcome_polarity_flip"
RESIDUAL_ORDINAL = "ordinal_outside_token_class"


def outcome_polarity_marks(text: str) -> tuple[str, ...]:
    """Outcome-polarity markers in ``text``, de-duplicated and sorted.

    Shape-based on purpose. A bare ``pass`` or ``fail`` occurs constantly in
    ordinary prose ("acceptance pass", "the run did not fail"), so counting
    those would report flips that are not flips. The two shapes here are the
    two the corpus actually uses to record an outcome.
    """

    marks = {f"completed({value})" for value in _POLARITY_COMPLETED_RE.findall(text)}
    marks |= {f"OUTCOME {value}" for value in _POLARITY_OUTCOME_RE.findall(text)}
    return tuple(sorted(marks))


def ordinal_marks(text: str) -> tuple[str, ...]:
    """``#N`` ordinals in ``text``, de-duplicated and sorted."""

    return tuple(sorted({f"#{value}" for value in _ORDINAL_RE.findall(text)}))


def _polarity_words(marks: Iterable[str]) -> set[str]:
    """The bare polarity words a marker set carries: ``PASS``, ``fail``, ..."""

    words: set[str] = set()
    for mark in marks:
        if mark.startswith("completed("):
            words.add(mark[len("completed(") : -1])
        elif mark.startswith("OUTCOME "):
            words.add(mark[len("OUTCOME ") :])
    return words


def _residual_hits(pair: PairRecord) -> dict[str, dict[str, Any]]:
    """Which residual classes this pair carries, and the proof of invisibility.

    A class "applies" when the two texts differ in that class's marks. The
    invisibility check runs the shipped ``extract_identifiers`` over both full
    texts and asserts the differing marks' own tokens are absent from both
    identifier sets -- that is what makes the difference structurally
    unreachable by the veto rather than merely unreached in this corpus.

    Everything here reads the FULL texts, never the previews: a ``#N`` past
    the preview cutoff is still a difference the drain would hide, and
    scanning the preview would have silently undercounted the long records --
    several pairs in this corpus run to thousands of characters.
    """

    hits: dict[str, dict[str, Any]] = {}
    bearer_text = pair.bearer_full_text
    candidate_text = pair.candidate_full_text
    bearer_tokens = set(extract_identifiers(bearer_text))
    candidate_tokens = set(extract_identifiers(candidate_text))
    tokens = bearer_tokens | candidate_tokens

    bearer_polarity = outcome_polarity_marks(bearer_text)
    candidate_polarity = outcome_polarity_marks(candidate_text)
    # A FLIP, not an asymmetry: both texts must record an outcome and the
    # recorded polarities must differ. One text carrying a marker the other
    # simply lacks is two differently-shaped records, which the rest of the
    # report already covers -- counting it here would inflate this class with
    # pairs it is not about.
    bearer_words = _polarity_words(bearer_polarity)
    candidate_words = _polarity_words(candidate_polarity)
    if bearer_words and candidate_words and bearer_words != candidate_words:
        differing = sorted(set(bearer_polarity) ^ set(candidate_polarity))
        words = bearer_words ^ candidate_words
        hits[RESIDUAL_OUTCOME_POLARITY] = {
            "bearer_marks": list(bearer_polarity),
            "candidate_marks": list(candidate_polarity),
            "differing_marks": differing,
            # The words that make the difference, and whether the extractor
            # can see any of them. Empty intersection == invisible.
            "words_visible_to_extractor": sorted(words & tokens),
            "invisible": not (words & tokens),
        }

    bearer_ordinals = ordinal_marks(bearer_text)
    candidate_ordinals = ordinal_marks(candidate_text)
    if set(bearer_ordinals) != set(candidate_ordinals):
        differing = sorted(set(bearer_ordinals) ^ set(candidate_ordinals))
        digits = {mark.lstrip("#") for mark in differing}
        hits[RESIDUAL_ORDINAL] = {
            "bearer_ordinals": list(bearer_ordinals),
            "candidate_ordinals": list(candidate_ordinals),
            "differing_ordinals": differing,
            # ``#1234`` extracts as ``1234`` and IS an identifier (four
            # characters with a digit); ``#2`` and ``#12`` are not. Reporting
            # which is which keeps the class claim exact.
            "digits_visible_to_extractor": sorted(digits & tokens),
            "invisible": not (digits & tokens),
        }
    return hits


def residual_classes(pairs: Sequence[PairRecord]) -> dict[str, Any]:
    """Per-class, per-band counts for the differences the veto cannot see."""

    definitions = (
        (
            RESIDUAL_OUTCOME_POLARITY,
            "Outcome-polarity flips",
            "`completed (PASS)` against `completed (FAIL)`, `OUTCOME pass` "
            "against `OUTCOME fail`.",
            "`extract_identifiers` deliberately treats an all-caps word and a "
            "bare lowercase word as prose, not as a name -- the hyphen, "
            "underscore, slash, dot, ULID and digest rules are what make a "
            "token an identifier and `PASS`, `FAIL`, `pass`, `fail` match "
            "none of them. No identifier rule, however tightened, can see the "
            "flip: it is not an identifier difference.",
            "the two outcome-record shapes this corpus uses, "
            "`completed (PASS|FAIL)` and `OUTCOME pass|fail`",
        ),
        (
            RESIDUAL_ORDINAL,
            "Ordinals outside the token character class",
            "`Latency benchmark probe #2` against `Latency benchmark probe`.",
            "`#` is absent from the token character class "
            "(`[\\w~/][\\w./:~-]*`), so `#2` is tokenized as the bare `2` -- "
            "one character, below the two-character floor, and dropped. `#12` "
            "survives tokenization but carries no separator, no dot and only "
            "two characters, so it fails the weak-identifier floor of four "
            "too. From `#1234` on the digits ARE an identifier, which is why "
            "this class is reported with the visible/invisible split rather "
            "than as a blanket claim.",
            "`#` followed by digits",
        ),
    )

    classes: list[dict[str, Any]] = []
    for key, title, example, why, detector in definitions:
        per_band: dict[str, Any] = {}
        all_hits: list[PairRecord] = []
        invisibility_holds = True
        for band in (BAND_HIGH, BAND_CONTEXT):
            in_band = [pair for pair in pairs if pair.band == band]
            survivor_arm = ARM_HIGH_VETO_ON if band == BAND_HIGH else ARM_CONTEXT_VETO_ON
            matched: list[dict[str, Any]] = []
            for pair in in_band:
                hit = _residual_hits(pair).get(key)
                if hit is None:
                    continue
                all_hits.append(pair)
                invisibility_holds = invisibility_holds and bool(hit["invisible"])
                matched.append(
                    {
                        "pair_id": pair.pair_id,
                        "cosine": pair.cosine,
                        "scope": pair.scope,
                        "verdict": pair.verdict,
                        "identifier_veto_blocks": pair.identifier_veto_blocks,
                        # What the veto blocked on instead, when it blocked --
                        # the difference between "caught" and "caught for an
                        # unrelated reason".
                        "blocked_on": list(pair.identifiers_lost[:4]),
                        "survives_veto": survivor_arm in pair.arms,
                        "evidence": hit,
                    }
                )
            surviving = [entry for entry in matched if entry["survives_veto"]]
            per_band[band] = {
                "present": bool(matched),
                "pairs": len(matched),
                "pair_ids": [entry["pair_id"] for entry in matched],
                "verdict_different_facts": len(
                    [e for e in matched if e["verdict"] == VERDICT_DIFFERENT]
                ),
                # Blocked, but never ON this class -- the class is invisible,
                # so any block is on some unrelated token that happened to
                # differ too. The artifact says so rather than letting the
                # block read as the veto catching the flip.
                "blocked_by_veto_on_an_unrelated_token": len(
                    [e for e in matched if e["identifier_veto_blocks"]]
                ),
                "surviving_veto": len(surviving),
                "surviving_different_facts": len(
                    [e for e in surviving if e["verdict"] == VERDICT_DIFFERENT]
                ),
                "matches": matched,
            }
        classes.append(
            {
                "key": key,
                "title": title,
                "example": example,
                "why_the_veto_cannot_see_it": why,
                "detector": detector,
                "total_pairs": len(all_hits),
                "invisibility_verified": invisibility_holds,
                "bands": per_band,
            }
        )
    return {
        "note": (
            "The identifier veto compares identifier TOKENS. A pair that "
            "differs in something that is not an identifier token is "
            "invisible to it by construction. These are the classes of that "
            "kind present in this corpus, each counted from this same run. "
            "'Zero different_facts pairs measured at >=0.99' is a statement "
            "about the pairs this corpus produced, NOT a claim that no "
            "different-facts pair can get through."
        ),
        "classes": classes,
    }


# --------------------------------------------------------------------------
# The substrate is read-only, and that is checked after the fact
# --------------------------------------------------------------------------


def substrate_fingerprint(path: Path) -> dict[str, Any]:
    """Enough of the backup file's state to prove the run did not touch it."""

    resolved = path.expanduser()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "mtime": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(
            timespec="microseconds"
        ),
        "wal_present": Path(f"{resolved}-wal").exists(),
        "shm_present": Path(f"{resolved}-shm").exists(),
    }


def verify_substrate_untouched(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Raise unless the backup is byte-identical and has no journal beside it.

    ``mode=ro`` is a promise SQLite makes; this is the check that it kept it.
    A changed mtime or a ``-wal`` appearing next to the backup would both mean
    a writable connection was opened somewhere, which is the one thing this
    script may not do.
    """

    problems: list[str] = []
    for field_name in ("size_bytes", "mtime_ns"):
        if before[field_name] != after[field_name]:
            problems.append(
                f"{field_name} changed: {before[field_name]} -> {after[field_name]}"
            )
    for sidecar in ("wal_present", "shm_present"):
        if after[sidecar]:
            problems.append(f"{sidecar} after the run: a journal was created")
    if problems:
        raise SimulationError(
            "the backup was not left untouched: " + "; ".join(problems)
        )
    return {
        "before": dict(before),
        "after": dict(after),
        "mtime_unchanged": True,
        "wal_absent_after_run": True,
        "checked": "size, mtime_ns, and the absence of -wal/-shm beside the backup",
    }


# --------------------------------------------------------------------------
# The flag grep
# --------------------------------------------------------------------------


def grep_drain_flag(root: Path) -> dict[str, Any]:
    """Prove by grep that nothing in the tree turns the drain on.

    "Mentions the name" is not "enables it": the constant lives in
    ``retrieval.py``, the tests set it, this script documents it, and the
    artifact quotes it. What would be an enablement is an ASSIGNMENT to a
    truthy value outside a test -- ``LM_DRAIN_NEAR_DUP_SUPERSEDES=1`` in a
    shell script, a dotenv file, a systemd unit, a compose file, or a source
    default. Those are what this looks for.
    """

    hits = subprocess.run(
        ["git", "grep", "-n", "-I", DRAIN_NEAR_DUP_ENV],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line for line in hits.stdout.splitlines() if line.strip()]
    assignment = re.compile(
        rf"{re.escape(DRAIN_NEAR_DUP_ENV)}\s*(=|:)\s*[\"']?(1|true|yes|on)\b",
        re.IGNORECASE,
    )
    enabling: list[str] = []
    for line in lines:
        path, _, text = line.partition(":")
        if path.startswith("tests/"):
            # A test that sets the gate is the gate being TESTED, not shipped.
            continue
        if path == f"scripts/{PROGRAM}" or path.startswith("artifacts/"):
            # This script and its own artifact quote the line for the operator.
            continue
        if assignment.search(text):
            enabling.append(line)
    return {
        "pattern": DRAIN_NEAR_DUP_ENV,
        "command": f"git grep -n -I {DRAIN_NEAR_DUP_ENV}",
        "mentions": len(lines),
        "mentioning_files": sorted({line.partition(":")[0] for line in lines}),
        "enabling_assignments_outside_tests": enabling,
        "enabled_anywhere": bool(enabling),
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _short(node_id: str) -> str:
    return node_id[-8:] if len(node_id) > 8 else node_id


def _cell(text: str, limit: int = 90) -> str:
    flat = " ".join(text.split())
    if len(flat) > limit:
        flat = flat[: limit - 1] + "…"
    return flat.replace("|", "\\|")


def _pair_table(pairs: Sequence[PairRecord], survivor_arm: str) -> str:
    """One row per pair. Ids are truncated for width; the JSON carries them whole."""

    if not pairs:
        return "_No pair in this band._\n"
    lines = [
        "| pair (bearer → candidate) | scope | cosine | chars b/c | "
        "identifiers the collapse would hide | veto | survives | verdict | read as |",
        "| --- | --- | ---: | ---: | --- | :-: | :-: | --- | --- |",
    ]
    for pair in pairs:
        lost = ", ".join(f"`{token}`" for token in pair.identifiers_lost[:3])
        if len(pair.identifiers_lost) > 3:
            lost += f" _(+{len(pair.identifiers_lost) - 3})_"
        lines.append(
            "| `{bearer}` → `{candidate}` | {scope} | {cosine:.5f} | "
            "{bc}/{cc} | {lost} | {veto} | {survives} | {verdict} | {note} |".format(
                bearer=_short(pair.bearer_id),
                candidate=_short(pair.candidate_id),
                scope=pair.scope,
                cosine=pair.cosine,
                bc=pair.bearer_chars,
                cc=pair.candidate_chars,
                lost=_cell(lost, 70) or "—",
                veto="**blocks**" if pair.identifier_veto_blocks else "passes",
                survives="**yes**" if survivor_arm in pair.arms else "no",
                verdict=pair.verdict or "**UNREAD**",
                note=_cell(pair.verdict_note or "", 150),
            )
        )
    return "\n".join(lines) + "\n"


def _pair_detail(pair: PairRecord) -> str:
    lost = ", ".join(f"`{token}`" for token in pair.identifiers_lost) or "_(none)_"
    bearer_only = (
        ", ".join(f"`{token}`" for token in pair.identifiers_bearer_only[:10]) or "_(none)_"
    )
    return (
        f"#### `{pair.bearer_id}` → `{pair.candidate_id}` "
        f"— cosine {pair.cosine:.5f}, {pair.scope}\n\n"
        f"- verdict: **{pair.verdict or 'UNREAD'}** — {pair.verdict_note or ''}\n"
        f"- identifier veto: **{'blocks' if pair.identifier_veto_blocks else 'passes'}**; "
        f"candidate identifiers absent from the bearer: {lost}\n"
        f"- bearer-only identifiers (kept by the collapse, shown for symmetry): {bearer_only}\n"
        f"- survives the veto at 0.99: "
        f"**{'yes' if ARM_HIGH_VETO_ON in pair.arms else 'no'}**\n\n"
        f"BEARER ({pair.bearer_chars} chars"
        f"{', preview' if pair.bearer_text_truncated else ''}):\n\n"
        f"```\n{pair.bearer_text}\n```\n\n"
        f"CANDIDATE ({pair.candidate_chars} chars"
        f"{', preview' if pair.candidate_text_truncated else ''}):\n\n"
        f"```\n{pair.candidate_text}\n```\n"
    )


def _residual_section(residual: Mapping[str, Any]) -> str:
    """The section that stops "zero measured" from being read as "zero possible"."""

    out: list[str] = []
    out.append("\n## What the identifier veto structurally cannot see\n")
    out.append(residual["note"] + "\n")
    for entry in residual["classes"]:
        high = entry["bands"][BAND_HIGH]
        context = entry["bands"][BAND_CONTEXT]
        out.append(f"\n### {entry['title']}\n")
        out.append(
            f"Example: {entry['example']}\n\n"
            f"Why the veto cannot see it: {entry['why_the_veto_cannot_see_it']}\n\n"
            f"Detected here by scanning both texts of every pair for "
            f"{entry['detector']}, then checking the differing marks against "
            f"the shipped `extract_identifiers` over the same texts. "
            f"Invisibility held for every match in this run: "
            f"**{'yes' if entry['invisibility_verified'] else 'NO — see matches'}**.\n\n"
        )
        out.append(
            "| band | present | pairs | read as `different_facts` | blocked "
            "(on an unrelated token) | survives the veto | surviving AND "
            "`different_facts` |\n"
            "| --- | :-: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for label, band in (
            (">= 0.99 — the band proposed for automation", high),
            ("0.95-0.99 — context", context),
        ):
            out.append(
                f"| {label} | {'**yes**' if band['present'] else 'no'} | "
                f"{band['pairs']} | {band['verdict_different_facts']} | "
                f"{band['blocked_by_veto_on_an_unrelated_token']} | "
                f"{band['surviving_veto']} | "
                f"**{band['surviving_different_facts']}** |\n"
            )
        out.append("\n")
        for label, band in ((BAND_HIGH, high), (BAND_CONTEXT, context)):
            if not band["matches"]:
                out.append(f"_No pair of this class in `{label}`._\n")
                continue
            out.append(f"In `{label}`:\n\n")
            for match in band["matches"]:
                evidence = match["evidence"]
                differing = evidence.get("differing_marks") or evidence.get(
                    "differing_ordinals", []
                )
                fate = (
                    "**survives**"
                    if match["survives_veto"]
                    else (
                        "blocked, on "
                        + ", ".join(f"`{token}`" for token in match["blocked_on"])
                        + " — an unrelated token, not this difference"
                        if match["blocked_on"]
                        else "not produced in the veto-on arm"
                    )
                )
                out.append(
                    f"- `{match['pair_id']}` (cosine {match['cosine']:.5f}, "
                    f"{match['scope']}) — {', '.join('`' + m + '`' for m in differing)}"
                    f"; read as `{match['verdict']}`; {fate}\n"
                )
            out.append("\n")
    return "".join(out)


def render_markdown(report: Mapping[str, Any], pairs: Sequence[PairRecord]) -> str:
    high = report["bands"][BAND_HIGH]
    context = report["bands"][BAND_CONTEXT]
    copy = report["copy"]
    coverage = report["coverage"]
    arms = {arm["name"]: arm for arm in report["arms"]}
    high_pairs = [pair for pair in pairs if pair.band == BAND_HIGH]
    context_pairs = [pair for pair in pairs if pair.band == BAND_CONTEXT]
    different_pairs = [pair for pair in pairs if pair.verdict == VERDICT_DIFFERENT]

    out: list[str] = []
    out.append("# Drain near-duplicate simulation on the pre-hygiene backup\n")
    out.append(
        f"Generated by `scripts/{PROGRAM}` at {report['generated_at']}. "
        "Every pair below is a pair the SHIPPED "
        "`MemoryRecallService._supersede_drained_near_dups` decided to write, "
        "intercepted before it reached storage. No reimplementation of the "
        "drain's rules appears anywhere in this report.\n"
    )

    out.append("\n## The answer\n")
    verdict_line = (
        "**The >=0.99 band is automatable.**"
        if high["automatable"]
        else "**The >=0.99 band is NOT automatable as measured.**"
    )
    out.append(
        f"{verdict_line} At threshold 0.99 the drain's own semantics would "
        f"collapse **{high['would_collapse_pairs_veto_off']}** pairs with the "
        f"identifier veto off. Reading both texts of each: "
        f"**{high['verdict_verbatim_repeat']}** are `verbatim_repeat`, "
        f"**{high['verdict_different_facts']}** are `different_facts`. "
        f"The identifier veto blocks **{high['different_facts_blocked_by_veto']}** "
        f"of those `different_facts` pairs, leaving "
        f"**{high['different_facts_surviving_veto']}** surviving — the number "
        "that has to be zero.\n"
    )
    if high["verbatim_repeats_blocked_by_veto"]:
        out.append(
            f"Price of the veto in the same band: it also blocks "
            f"**{high['verbatim_repeats_blocked_by_veto']}** pairs that reading "
            f"says are honest repeats. Net, the drain writes "
            f"**{high['surviving_pairs_after_veto']}** edges instead of "
            f"**{high['would_collapse_pairs_veto_off']}**.\n"
        )
    else:
        out.append(
            f"Price of the veto in the same band: **none** — every pair it "
            f"blocks here was read as `different_facts`. The drain writes "
            f"**{high['surviving_pairs_after_veto']}** edges instead of "
            f"**{high['would_collapse_pairs_veto_off']}**. (The 0.95-0.99 band "
            f"below is where the veto starts costing honest collapses: "
            f"**{context['verbatim_repeats_blocked_by_veto']}** there.)\n"
        )
    residual_high_total = sum(
        entry["bands"][BAND_HIGH]["pairs"]
        for entry in report["residual_classes"]["classes"]
    )
    out.append(
        f"\nWhat that number does NOT say: the veto compares identifier "
        f"tokens, so a pair differing in anything else is invisible to it by "
        f"construction. **{residual_high_total}** pair(s) in this band "
        f"already carry such a difference — see *What the identifier veto "
        f"structurally cannot see* below, which names each class, counts it "
        f"in both bands, and says whether the veto stopped the pair for some "
        f"unrelated reason.\n"
    )
    if high["different_facts_surviving_veto"]:
        by_id = {pair.pair_id: pair for pair in pairs}
        out.append(
            "\n### The survivors, each read\n\n"
            "A pair survives the veto when the candidate names no identifier "
            "token the bearer lacks — so for each of these the collapse hides "
            "a difference that is not an identifier at all. What that "
            "difference is, per pair:\n\n"
        )
        for pair in high_pairs:
            if pair.verdict != VERDICT_DIFFERENT or ARM_HIGH_VETO_ON not in pair.arms:
                continue
            mirror = by_id.get(f"{pair.candidate_id}->{pair.bearer_id}")
            out.append(
                f"- `{pair.bearer_id}` → `{pair.candidate_id}` "
                f"(cosine {pair.cosine:.5f}, {pair.scope}) — collapses in arms "
                f"{', '.join('`' + name + '`' for name in sorted(pair.arms))}. "
            )
            if mirror is None:
                out.append(
                    "**No mirror pair exists in any arm**: this direction is "
                    "the only one the candidate ordering ever produces, and "
                    "the veto never gets a chance to see the collapse from the "
                    "side where the identifier is missing. Turning the veto on "
                    "changes nothing about this pair.\n"
                )
            else:
                out.append(
                    f"Its mirror `{mirror.bearer_id}` → `{mirror.candidate_id}` "
                    f"IS blocked by the veto, and runs in arms "
                    f"{', '.join('`' + name + '`' for name in sorted(mirror.arms))}. "
                    "Read the two together: turning the veto ON did not prevent "
                    "this collapse, it **reversed** it. With the veto off the "
                    "drain demotes one node; with the veto on it demotes the "
                    "other. Either way one of two distinct tree nodes loses its "
                    "recorded outcome to the other.\n"
                )
            out.append(f"  - {pair.verdict_note}\n")
    out.append(
        "\n**This node did not enable the drain.** "
        f"`{DRAIN_NEAR_DUP_ENV}` is set nowhere in the tree "
        f"({report['flag_grep']['mentions']} mentions, "
        f"{len(report['flag_grep']['enabling_assignments_outside_tests'])} enabling "
        "assignments outside tests); the operator handoff is at the bottom.\n"
    )

    out.append("\n## Substrate\n")
    out.append(
        f"- backup: `{copy['substrate']}` ({copy['substrate_bytes']:,} bytes), "
        f"opened {copy['substrate_opened']}\n"
        f"- working copy: `{copy['copy']}`, sha256 `{copy['copy_sha256']}`\n"
        f"- corpus: {copy['nodes']:,} nodes, {copy['active_nodes']:,} active, "
        f"{copy['active_traces']:,} active traces, {copy['chunk_rows']:,} chunk rows, "
        f"{copy['connections']:,} connections ({copy['supersedes_edges']:,} supersedes)\n"
        f"- the live `global.sqlite3` and the alt database were never opened; "
        f"`refuse_live_database` raises on their names\n"
    )

    out.append("\n## How the pairs were produced\n")
    out.append(report["method"] + "\n")

    out.append("\n### How candidates were offered\n")
    out.append(report["candidate_offering"] + "\n")
    out.append(
        f"Coverage: **{coverage['active_traces']:,}** active traces exist, "
        f"**{coverage['traces_with_pooled_vector']:,}** of them have a usable "
        f"mean-pooled vector and could therefore be compared at all; "
        f"**{coverage['traces_without_pooled_vector']:,}** have no chunk rows "
        "(or rows of disagreeing width) and are invisible to this pass by "
        "construction — `near_dup` treats an absent vector as *never collapsed*, "
        "never as similar-to-everything. The backup was NOT warm-drained first: "
        "chunking it would have written to the copy and changed the corpus "
        "under the measurement.\n"
    )

    density = report.get("neighbour_density") or {}
    if density.get("available"):
        out.append("\n### The denominator\n")
        out.append(
            "How many near-duplicates actually exist, before any veto: active "
            f"traces with another active trace of the same scope above "
            f"**0.95: {density['gt_0_95']:,}**, above **0.98: "
            f"{density['gt_0_98']:,}**, above **0.99: {density['gt_0_99']:,}** "
            "(mean-pooled over the same cached matrix the pass compares "
            "with). Read the pair counts below against these: the pass "
            "proposes far fewer pairs than there are near-duplicates, and "
            "that gap is the shipped vetoes — provenance, corrections, "
            "already-superseded, level/scope, length — doing their job before "
            "the identifier veto is ever consulted.\n"
        )

    out.append("\n### Arms\n")
    out.append(
        "| arm | threshold | identifier veto | pairs | scopes | arrivals offered | seconds |\n"
        "| --- | ---: | :-: | ---: | ---: | ---: | ---: |\n"
    )
    for name, _threshold, _veto in ARMS:
        arm = arms[name]
        out.append(
            f"| `{name}` | {arm['threshold']} | "
            f"{'on' if arm['identifier_veto'] else 'off'} | {arm['pairs']} | "
            f"{arm['scopes_run']} | {arm['candidates_offered']:,} | "
            f"{arm['seconds']:.1f} |\n"
        )

    out.append("\n## Band >= 0.99 — the band proposed for automation\n")
    out.append(
        "Pairs the drain would write at its shipped threshold with the veto "
        "OFF. `survives veto` is membership in the `t0.99-veto-on` arm, i.e. "
        "what the shipped default actually writes.\n\n"
    )
    out.append(_pair_table(high_pairs, ARM_HIGH_VETO_ON))
    out.append("\n### Every pair in this band, read\n\n")
    for pair in high_pairs:
        out.append(_pair_detail(pair) + "\n")

    out.append("\n## Band 0.95-0.99 — context, not a proposal\n")
    out.append(
        "What the 0.99 floor buys. These pairs are produced by the same pass "
        "with `LM_DRAIN_NEAR_DUP_COSINE=0.95`; nothing in the acceptance "
        "depends on them, and no operator is being offered this band.\n\n"
    )
    out.append(
        f"- would collapse (veto off): **{context['would_collapse_pairs_veto_off']}** pairs\n"
        f"- read as `verbatim_repeat`: **{context['verdict_verbatim_repeat']}**; "
        f"as `different_facts`: **{context['verdict_different_facts']}**\n"
        f"- `different_facts` the veto blocks: "
        f"**{context['different_facts_blocked_by_veto']}**; surviving: "
        f"**{context['different_facts_surviving_veto']}**\n\n"
    )
    out.append(_pair_table(context_pairs, ARM_CONTEXT_VETO_ON))

    survivors = [
        pair
        for pair in pairs
        if pair.verdict == VERDICT_DIFFERENT
        and (
            (pair.band == BAND_HIGH and ARM_HIGH_VETO_ON in pair.arms)
            or (pair.band == BAND_CONTEXT and ARM_CONTEXT_VETO_ON in pair.arms)
        )
    ]
    if survivors:
        out.append("\n## Every `different_facts` pair that SURVIVES the veto, in full\n")
        out.append(
            f"{len(survivors)} pairs across both bands "
            f"({len([p for p in survivors if p.band == BAND_HIGH])} at >=0.99). "
            "These are the collapses the shipped default would actually write "
            "that reading says join two facts. Full texts of every other pair "
            "are in `drain-simulation.json`; the tables above carry every "
            "pair's verdict and the one-line reason for it.\n\n"
        )
        for pair in survivors:
            out.append(_pair_detail(pair) + "\n")
    out.append(
        f"\n_Read across both bands: {len(different_pairs)} of {len(pairs)} pairs "
        "are `different_facts`. The tables above give each one its verdict and "
        "reason; `drain-simulation.json` carries both texts of all of them._\n"
    )

    out.append(_residual_section(report["residual_classes"]))

    out.append("\n## Nothing was written\n")
    safety = report["safety"]
    integrity = safety["substrate_integrity"]
    out.append(
        f"- `connections` rows before the run: **{safety['connections_before']:,}**; "
        f"after all four arms: **{safety['connections_after']:,}**; "
        f"delta **{safety['connections_delta']}**\n"
        f"- `store.create_connection` was an instance-level recorder for every "
        f"arm; **{safety['edges_intercepted']}** calls were intercepted and "
        f"**0** reached storage\n"
        f"- a SQLite authorizer DENIED insert/update/delete on `connections` "
        f"for the whole run, so an escape would have raised rather than landed\n"
        f"- the backup itself: mtime `{integrity['before']['mtime']}` before "
        f"the run and `{integrity['after']['mtime']}` after, "
        f"{integrity['before']['size_bytes']:,} bytes both times, and no "
        f"`-wal` or `-shm` beside it after the run — checked, not assumed\n"
        f"- `{DRAIN_NEAR_DUP_ENV}` was "
        f"`{safety['drain_gate_in_process']}` in the simulating process\n"
    )

    out.append("\n## Operator handoff\n")
    residual_high = [
        entry
        for entry in report["residual_classes"]["classes"]
        if entry["bands"][BAND_HIGH]["present"]
    ]
    if high["automatable"]:
        out.append(
            f"**The measurement clears the band.** Of the "
            f"{high['would_collapse_pairs_veto_off']} pairs the drain would "
            f"collapse at 0.99 with the veto off, "
            f"{high['verdict_different_facts']} were read as "
            f"`different_facts` and the veto blocks every one; the "
            f"{high['surviving_pairs_after_veto']} edges the shipped default "
            "would actually write were all read as honest repeats.\n\n"
            "**Read the caveat with the number.** Zero `different_facts` "
            "pairs SURVIVED here is a measurement over the pairs this corpus "
            "produced — it is not a claim that no different-facts pair can "
            "get through. The veto compares identifier tokens and nothing "
            "else, so every class listed under *What the identifier veto "
            "structurally cannot see* remains open"
            + (
                ": "
                + ", ".join(
                    f"{entry['title'].lower()} "
                    f"({entry['bands'][BAND_HIGH]['pairs']} pair(s) already at "
                    f">=0.99 in this run)"
                    for entry in residual_high
                )
                + ". "
                if residual_high
                else ". "
            )
            + "An outcome-polarity flip with equal durations, or a `#N` "
            "ordinal on otherwise identical text, would collapse at 0.99 "
            "today.\n\n"
        )
    if not high["automatable"]:
        out.append(
            "**Recommendation: do not set this yet.** The band this artifact "
            f"was asked to clear does not clear: "
            f"{high['different_facts_surviving_veto']} of the "
            f"{high['surviving_pairs_after_veto']} edges the shipped default "
            "would write at 0.99 join two different tree nodes. The env line "
            "is documented below because the goal asks for it to be, not "
            "because the measurement supports using it.\n\n"
        )
    out.append(
        "This node produced evidence and did NOT flip the flag. The line an "
        "operator would add to the LM server's environment is:\n\n"
        f"```sh\n{OPERATOR_ENV_BLOCK}```\n\n"
        f"`{DRAIN_NEAR_DUP_ENV}` alone is the gate: unset, `0`, or anything "
        "unrecognized leaves the drain writing chunks and nothing else. "
        f"`{DRAIN_NEAR_DUP_COSINE_ENV}` defaults to "
        f"{DEFAULT_DRAIN_NEAR_DUP_COSINE} — the band this artifact classified — "
        "and set to `0` it disables the collapse without unsetting the gate. "
        f"`{IDENTIFIER_VETO_ENV}` is default-ON and does most of the work: it "
        f"blocks {high['different_facts_blocked_by_veto']} of the "
        f"{high['verdict_different_facts']} `different_facts` pairs at 0.99, "
        "and setting it to `0` while the gate is on re-admits all of them. "
        + (
            "What it does NOT do is make the band safe on its own — see the "
            "survivors above, which it passes.\n\n"
            if high["different_facts_surviving_veto"]
            else "What it does NOT do is make the band safe on its own: it "
            "blocks identifier differences, and the residual classes named "
            "above are exactly the differences that are not identifier "
            "differences.\n\n"
        )
        + "Edges the pass writes are `supersedes` rows tagged "
        f"`metadata['kind'] = '{DRAIN_NEAR_DUP_KIND}'`, which is how a run is "
        "reverted as a group. Nothing is deleted and nothing is decayed: a "
        "demoted node keeps its row, its chunks and its text and stays "
        "reachable through `memory_lookup`.\n"
    )
    out.append("\nGrep evidence that the flag is enabled nowhere:\n\n")
    out.append(f"```\n$ {report['flag_grep']['command']}\n")
    for path in report["flag_grep"]["mentioning_files"]:
        out.append(f"{path}\n")
    out.append(
        f"\nenabling assignments outside tests: "
        f"{report['flag_grep']['enabling_assignments_outside_tests'] or 'none'}\n```\n"
    )
    return "".join(out)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


METHOD = """\
1. The backup is opened once, `mode=ro`, and `VACUUM INTO` produces a working
   copy under the workdir. The live `global.sqlite3` is never opened.
2. A real `MemoryStore` is built over the copy and a real
   `MemoryRecallService` over that store.
3. `store.create_connection` is replaced, on the instance, by a recorder, and a
   SQLite authorizer denies writes to `connections`.
4. Per scope, in descending order of active-trace count:
   `service._supersede_drained_near_dups(scope, nodes, service._scope_chunk_index(scope, service._chunk_corpus_revision()))`.
   The pass's returned pair list is compared against what the recorder saw, per
   scope, so an unrecorded write would abort the run.
5. Four arms differing only in `LM_DRAIN_NEAR_DUP_COSINE` and
   `LM_NEAR_DUP_IDENTIFIER_VETO`, each with a fresh service.

Because the pass is called directly, every veto it applies is the shipped one:
the `source_traces` provenance guard, the corrections guard, the
already-superseded guard on candidate and bearer, the level/scope match, the
length guard and the cosine threshold inside `near_dup.build_duplicate_map`,
and the direction in which the edge points. This report restates none of them.

One documented consequence of running the pass per scope without writing: the
real pass reads `_supersedes_sets()` once per call, so within a scope the
simulation is exact. Across scopes, a real run would see the edges earlier
scopes wrote; the simulation does not. That cannot change a pair, because every
node in a pair is in the pair's own scope (`bearer.scope != scope` is a veto),
so edges from another scope name nodes this scope's pass never looks at.
"""

CANDIDATE_OFFERING = """\
In production `_supersede_drained_near_dups` receives `freshly_chunked[scope]`:
the nodes one recall just gave vectors to, usually a handful. This simulation
offers **every active trace of the scope** as an arrival, which is the
maximal-coverage reading of "the near-duplicate backlog of a corpus that was
never drained" — every one of them is still unsorted, so every one of them is a
possible arrival.

That choice changes which node of a pair is the bearer, so it is stated rather
than assumed:

* the pass sorts its candidates **greatest id first** and offers them for
  demotion in that order. ULIDs sort chronologically, so with the whole scope
  offered at once, the LATER-created node of a pair is offered first and is the
  one demoted, and the EARLIER-created node bears. "The established node
  absorbs the arrival" therefore resolves here to "the earlier node absorbs the
  later one" — which is the same rule the production drain applies, evaluated
  over the whole backlog instead of over one recall's arrivals.
* `_best_pooled_match` defers arrivals: it looks for a bearer among the settled
  part of the scope first, and consults the drain's own arrivals only when the
  settled part offers nothing above the floor. With every active trace offered,
  the settled part is the non-trace and decayed nodes, so the nominee normally
  comes from the deferred pass — the plain arg-max over the scope, which is the
  same node a production-sized arrival set would have found. The exception is a
  concept or schema node scoring above the floor against a trace: this run
  nominates it and then drops the pair on the `bearer.level != node.level`
  veto, where a small-arrival run might have nominated a settled trace. That
  direction can only LOSE pairs from this report, never invent them.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument("--backup", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument(
        "--json", type=Path, default=REPO_ROOT / "artifacts/near-dup/drain-simulation.json"
    )
    parser.add_argument(
        "--md", type=Path, default=REPO_ROOT / "artifacts/near-dup/drain-simulation.md"
    )
    parser.add_argument("--preview-chars", type=int, default=DEFAULT_PREVIEW_CHARS)
    parser.add_argument(
        "--scope-limit",
        type=int,
        default=0,
        help="only the N scopes with the most active traces (smoke runs)",
    )
    parser.add_argument(
        "--scope",
        action="append",
        default=None,
        dest="only_scopes",
        help="restrict to these scopes (repeatable; smoke runs only)",
    )
    parser.add_argument(
        "--allow-missing-verdicts",
        action="store_true",
        help="write the artifact with unread pairs, marked incomplete (reading pass only)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    refuse_enabled_drain_flag()
    refuse_live_database(args.backup)
    refuse_live_directory(args.workdir, args.backup)

    workdir = args.workdir.expanduser()
    workdir.mkdir(parents=True, exist_ok=True)
    copy_path = workdir / "drain-sim.sqlite3"
    if not args.quiet:
        print(f"copying {args.backup} -> {copy_path} (read-only VACUUM INTO)", flush=True)
    substrate_before = substrate_fingerprint(args.backup)
    if substrate_before["wal_present"]:
        raise SimulationError(
            f"a -wal sits beside {args.backup}: the backup is not quiescent and "
            "this run refuses to read it"
        )
    copy_info = vacuum_copy(args.backup, copy_path)

    store = MemoryStore(copy_path)
    install_connection_write_guard(store)
    connections_before = int(
        store.connection.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
    )

    scopes = scopes_with_active_traces(store)
    if args.only_scopes:
        wanted = set(args.only_scopes)
        scopes = [scope for scope in scopes if scope in wanted]
    if args.scope_limit:
        scopes = scopes[: args.scope_limit]

    arm_results: list[ArmResult] = []
    for name, threshold, veto in ARMS:
        if not args.quiet:
            print(
                f"arm {name}: threshold={threshold} identifier_veto="
                f"{'on' if veto else 'off'}",
                flush=True,
            )
        arm_results.append(
            run_arm(store, name, threshold, veto, scopes, progress=not args.quiet)
        )
        if not args.quiet:
            print(
                f"  [{name}] done: {len(arm_results[-1].edges)} pairs in "
                f"{arm_results[-1].seconds:.1f}s",
                flush=True,
            )

    connections_after = int(
        store.connection.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
    )
    if connections_after != connections_before:
        raise SimulationError(
            "the simulation wrote to connections: "
            f"{connections_before} -> {connections_after}"
        )

    pairs = build_pairs(store, arm_results, preview_chars=args.preview_chars)
    coverage = pooled_vector_coverage(store, scopes)
    density = neighbour_density(MemoryRecallService(store), scopes, store)
    store.close()

    substrate_integrity = verify_substrate_untouched(
        substrate_before, substrate_fingerprint(args.backup)
    )

    unread = [pair.pair_id for pair in pairs if pair.verdict is None]
    if unread and not args.allow_missing_verdicts:
        _report_unread(pairs)
        raise SimulationError(
            f"{len(unread)} pair(s) have no verdict; every pair must be READ "
            "before the artifact is written. Re-run with "
            "--allow-missing-verdicts to dump them for reading."
        )

    report: dict[str, Any] = {
        "program": PROGRAM,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "purpose": (
            "Classify every pair the drain's real _supersede_drained_near_dups "
            "path would collapse on the pre-hygiene backup, so an operator can "
            "decide whether to set " + DRAIN_NEAR_DUP_ENV + "."
        ),
        "copy": copy_info,
        "method": METHOD,
        "candidate_offering": CANDIDATE_OFFERING,
        "coverage": coverage,
        "neighbour_density": density,
        "scopes_simulated": len(scopes),
        "arms": [
            {
                "name": arm.name,
                "threshold": arm.threshold,
                "identifier_veto": arm.identifier_veto,
                "env": {key: value for key, value in arm.env.items()},
                "pairs": len(arm.edges),
                "scopes_run": arm.scopes_run,
                "candidates_offered": arm.candidates_offered,
                "seconds": round(arm.seconds, 1),
                "edge_relation": sorted({edge.relation_type for edge in arm.edges}),
                "edge_kind": sorted(
                    {str(edge.metadata.get("kind")) for edge in arm.edges}
                ),
            }
            for arm in arm_results
        ],
        "bands": {
            BAND_HIGH: band_summary(pairs, BAND_HIGH, ARM_HIGH_VETO_ON),
            BAND_CONTEXT: band_summary(pairs, BAND_CONTEXT, ARM_CONTEXT_VETO_ON),
        },
        "residual_classes": residual_classes(pairs),
        "pairs": [pair.to_dict() for pair in pairs],
        "safety": {
            "connections_before": connections_before,
            "connections_after": connections_after,
            "connections_delta": connections_after - connections_before,
            "edges_intercepted": sum(len(arm.edges) for arm in arm_results),
            "create_connection": "instance-level recorder for every arm",
            "sqlite_authorizer": "DENY insert/update/delete on connections",
            "live_databases_opened": [],
            "substrate_integrity": substrate_integrity,
            "drain_gate_in_process": os.environ.get(DRAIN_NEAR_DUP_ENV, "<unset>"),
        },
        "operator_handoff": {
            "env_line": OPERATOR_ENV_LINE,
            "env_block": OPERATOR_ENV_BLOCK,
            "flag_flipped_by_this_node": False,
            "edge_kind": DRAIN_NEAR_DUP_KIND,
        },
        "flag_grep": grep_drain_flag(REPO_ROOT),
        "complete": not unread,
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(render_markdown(report, pairs), "utf-8")
    if not args.quiet:
        high = report["bands"][BAND_HIGH]
        print(
            f"\n{len(pairs)} distinct pairs; >=0.99 band: "
            f"{high['would_collapse_pairs_veto_off']} would-collapse, "
            f"{high['verdict_different_facts']} different_facts, "
            f"{high['different_facts_surviving_veto']} surviving the veto",
            flush=True,
        )
        print(f"wrote {args.json}\nwrote {args.md}", flush=True)
    return 0


def pooled_vector_coverage(store: MemoryStore, scopes: Sequence[str]) -> dict[str, Any]:
    """How much of the backlog could be compared at all.

    A trace with no chunk rows is absent from ``mean_pooled_vectors`` and is
    therefore never collapsed and never a bearer by similarity. Reporting the
    count is what stops "no pair here" from being read as "no repeat here".
    """

    placeholders = ",".join("?" for _ in scopes)
    total = int(
        store.connection.execute(
            f"SELECT COUNT(*) FROM nodes WHERE decayed = 0 AND level = 'trace' "
            f"AND scope IN ({placeholders})",
            tuple(scopes),
        ).fetchone()[0]
    )
    chunked = int(
        store.connection.execute(
            f"""
            SELECT COUNT(*) FROM nodes n
            WHERE n.decayed = 0 AND n.level = 'trace' AND n.scope IN ({placeholders})
              AND EXISTS (
                SELECT 1 FROM node_chunk_embeddings c WHERE c.node_id = n.id
              )
            """,
            tuple(scopes),
        ).fetchone()[0]
    )
    return {
        "active_traces": total,
        "traces_with_pooled_vector": chunked,
        "traces_without_pooled_vector": total - chunked,
    }


def neighbour_density(
    service: MemoryRecallService, scopes: Sequence[str], store: MemoryStore
) -> dict[str, Any]:
    """The raw near-duplicate population, before any veto: the denominator.

    Counts active traces that have ANOTHER active trace of the same scope above
    each threshold, using the same mean-pooled vectors the pass compares with
    (``_pooled_scope_vectors`` over the same cached index). This is arithmetic,
    not a second implementation of the drain's decision -- no veto, no
    direction, no bearer resolution -- and it exists so "9 would-collapse pairs"
    cannot be misread as "9 near-duplicates in the corpus". The gap between the
    two numbers IS the vetoes doing their job.
    """

    if _np is None:  # pragma: no cover - numpy is a normal runtime dep
        return {"available": False}
    counts = {"gt_0_95": 0, "gt_0_98": 0, "gt_0_99": 0}
    revision = service._chunk_corpus_revision()
    for scope in scopes:
        rows = store.connection.execute(
            "SELECT id FROM nodes WHERE decayed = 0 AND level = 'trace' AND scope = ?",
            (scope,),
        ).fetchall()
        trace_ids = {str(row["id"]) for row in rows}
        if len(trace_ids) < 2:
            continue
        pooled = _pooled_scope_vectors(service._scope_chunk_index(scope, revision))
        for block in pooled.values():
            keep = [
                row for row, node_id in enumerate(block.node_ids) if node_id in trace_ids
            ]
            if len(keep) < 2:
                continue
            matrix = _np.asarray(block.rows, dtype=_np.float32)[keep]
            best = _np.full(len(keep), -1.0, dtype=_np.float32)
            step = 512
            for start in range(0, len(keep), step):
                chunk = matrix[start : start + step]
                similarities = chunk @ matrix.T
                for offset in range(len(chunk)):
                    similarities[offset, start + offset] = -1.0
                best[start : start + step] = similarities.max(axis=1)
            counts["gt_0_95"] += int((best > 0.95).sum())
            counts["gt_0_98"] += int((best > 0.98).sum())
            counts["gt_0_99"] += int((best > 0.99).sum())
    return {
        "available": True,
        "definition": (
            "active traces with another active trace of the SAME scope above "
            "the threshold, mean-pooled over the same cached chunk matrix the "
            "pass uses; no veto of any kind applied"
        ),
        **counts,
    }


def _report_unread(pairs: Sequence[PairRecord]) -> None:
    unread = [pair for pair in pairs if pair.verdict is None]
    print(f"\n{len(unread)} pair(s) without a verdict:", file=sys.stderr)
    for pair in unread:
        print(f"  {pair.pair_id}  cos={pair.cosine:.5f}  {pair.scope}", file=sys.stderr)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SimulationError as error:
        print(f"{PROGRAM}: {error}", file=sys.stderr)
        raise SystemExit(2) from error
