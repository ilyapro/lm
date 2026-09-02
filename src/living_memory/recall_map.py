"""Recall map: what *else* memory holds for this task, as a handful of clusters.

A recall returns its top results and throws the rest of the ranked pool away.
The residual is not noise — it is the answer to a question the agent has not
asked yet, and the field measurement that motivates this module says the
question never gets asked: over 57 live recalls the median call sat at position
0.02 of the session, i.e. one start-of-session ritual and nothing after. The map
turns that single reliable recall into a *plan* for the mid-session ones: each
cluster carries a label, a count, a medoid example, a phrasing to ask with, and
a todo-plan line the agent can paste into its own plan.

Deterministic by construction
-----------------------------
No LLM call happens on this path and none may be added: the map is built inside
the live recall latency budget, and a model call there would cost more than the
recall it decorates. Labels come from a four-stage cascade, each stage claiming
what it can and handing the rest down:

1. **Structural context keys** — ``procedure_id``, ``lesson_kind``, ``type``,
   ``topic``, ``task_pattern``. The extraction stage already writes these, so
   most of a healthy corpus never reaches stage 2. The label *is* the key,
   normalized the way ``consolidation._normalize_trigger`` normalizes a
   procedure id, except where the raw key is an opaque hash — then the label
   comes from the group's medoid instead.
2. **Path collapse** — nodes carrying file paths become their subsystem, so
   ``src/living_memory/postsession/report.py`` reads as ``postsession`` rather
   than as five near-identical paths.
3. **Query anchors** — an anchor is a *past grounded query* (see
   ``query_anchors``), so an anchor covering a node supplies both the label and,
   verbatim, a proven way to ask for it. This is the only stage whose ask-hint
   is evidence rather than construction.
4. **Chunk-embedding fallback** — greedy leader clustering over the chunk
   vectors already in the store, labelled by c-TF-IDF against the FTS
   document-frequency index (``term_document_frequencies`` /
   ``fts_document_count``): terms frequent inside the cluster and rare in the
   corpus.

Identical input yields a byte-identical map. Every sort carries an explicit tie
break, the greedy pass walks the pool in rank order, and stage precedence is
fixed — a node with both a ``procedure_id`` and a file list is a structural
cluster member, never a path one.

Silence beats a word nobody can act on
--------------------------------------
The label *is* the interface. The first field window of this channel measured
what happens when it degenerates: 27 delivered injections, 0 consumptions, and
labels that had collapsed into single house words — ``public (73)``, ``test
(37)``, ``mmo (8)`` — while a control run of the same code over a corpus whose
labels read ``cleanup commit invariant`` was consumed on its first pass.

Two rules follow, and they work as a pair. Stages 1 and 2 *enrich*: a cluster
named after a structural value or a directory segment gets that name plus its
own most distinguishing terms (:meth:`RecallMapBuilder._enrich`), so the label
says something the corpus does not say everywhere. Then every label, from every
stage, is *gated*: :meth:`RecallMapBuilder._deliverable` withholds a cluster
whose best terms are collectively too common to distinguish anything
(:data:`LABEL_GATE_MIN_IC`). Enrichment is the rescue and the gate is the net —
after enrichment the gate should rarely fire, and when it does it is reporting a
cluster with nothing to say.

The gate is a corpus statistic, never a word list. "Generic" means generic to
the corpus this recall searched, measured against the same document-frequency
index the c-TF-IDF labeller already reads, so the rule transfers to a corpus
whose house vocabulary is different words instead of memorizing the ones a
field window happened to show.

And nothing leaves quietly. Every cluster the gate withholds and every cluster
the response budget drops is counted — and, as far as the budget allows, named
— in the payload's additive ``filtered`` key (:func:`_filter_block`), which
persists into ``recall_events.recall_map``. Without it a field reader cannot
tell "memory had nothing to say" from "the filter was too harsh", which are
opposite findings that both look like a short map.

Stability across sessions
-------------------------
A task that asks twice should see the same shape twice, or the map is furniture
rather than a plan. :class:`RecallMapBuilder` therefore caches the cluster
*structure* per ``(scope, normalized task)`` and revalidates it with the same
cheap monotonic probes ``retrieval`` uses for its chunk and anchor caches —
``(total_changes, PRAGMA data_version)`` in front of the chunk-table and
anchor-table revisions. Counts, medoids and membership refresh against the pool
in hand on every call; labels and ask-hints survive until the corpus moves.
There are no timers: a cache that expires on a clock would hand the same task a
different plan for no reason the agent can see.

Collapse when nobody follows it
-------------------------------
Delivered is not used. The preprompt push already taught this project that a
channel keeps costing its budget long after it stopped changing behaviour, so
the map is built with its own retreat: it measures whether the maps this key
already delivered were *followed*, and when three in a row were not
(:data:`CURTAIL_STREAK`) it stops occupying the channel and ships a marker
instead — ``curtailed: true`` with the zero-consumption streak, ~60 characters
where a map would have spent 700.

Consumption is measured, never assumed, from evidence the server already
writes (:meth:`RecallMapBuilder._curtailment`). A delivered cluster counts as
consumed when any of three probes fires after the delivery that offered it:

* **the ask was asked** — a later recall under the same key phrased a query
  covering :data:`CURTAIL_QUERY_OVERLAP` of the cluster's label or ask-hint
  tokens, under the tokenizer the grounding rail uses, so "followed the plan
  item" is the same measurement everywhere in this codebase;
* **the example was reached** — the cluster's medoid node was accessed after
  the delivery. ``retrieval`` stamps ``nodes.last_accessed`` on every result it
  delivers, so this fires the moment a later recall actually hands the agent
  the node the map pointed at;
* **the example was fetched** — a ``memory_lookup`` named the cluster's medoid
  by id inside the delivery's frozen outcome window. This is the only one of
  the three that no ranker can produce for itself: the two above both reduce to
  "a later recall returned this node again", which a hub schema satisfies
  forever without anybody reading it, while an id-fetch of one exact ULID is an
  act only a reader who was handed that ULID performs. The endogenous pair
  stays regardless — in curtail an ambiguity is resolved against collapsing, so
  a weak probe is evidence, not noise.

All three probes read tables the recall path already writes — a bounded window
over ``recall_events`` via ``MemoryStore.recent_recall_map_history``, one
primary-key batch over ``nodes``, and one over the delivery ledger that already
correlates lookups with their windows. Nothing new is written and no LLM is
asked.

The window read is not free, and pretending otherwise would be the wrong kind
of quiet: before the ``task_pattern`` column no index covered "this key's
deliveries that carried a map, newest first", so SQLite walked the scope's
whole partition and sorted — 8.5 ms on a 21k-event scope, measured, and rising
with it. ``idx_recall_events_scope_pattern_created`` narrows the pattern-keyed
form of that question to two bounded index ranges, but the read stays *gated*
rather than paid per recall, because the task-keyed form is unchanged and
because a narrower read is still a read: :class:`_CurtailMemo` counts the
offers this builder has made since it last read, and asks for a read only once
that count could reach the threshold. Its key is the map's key, both components
of it, or a per-turn client would seed a fresh memo every turn and pay on every
delivery the read the gate exists to avoid. The gate can delay a collapse and
can ask for a read it did not
need; it can never collapse a map by itself. What is left after the gate,
isolated by running ``scripts/recall_map_latency_bench.py`` twice over one
snapshot with the probe stubbed out of the second: +1.0 ms on the map stage's
median, which leaves the whole feature's pooled p95 overhead at 0.022 against
its pre-registered budget of 0.20.

The collapse is not a death sentence. Evidence lives in a window
(:data:`CURTAIL_HISTORY_LIMIT`), so the unread maps that silenced a key
eventually slide out from under the markers that replaced them and the key
offers again — 3 deliveries in 25 for a key that is never once followed.

The rule is unconditional. There is no environment flag and no opt-in: a
retreat that has to be switched on is a retreat nobody takes, and the protocol
channels forbid machinery an agent cannot rely on being there. Every ambiguity
in the evidence resolves *against* collapsing — an unreadable history, a
timestamp tie, a key whose deliveries this store cannot see all read as "used"
— because the cost of curtailing a map that was working is larger than the cost
of one more unread map.

How fast a collapsed key *returns* is the one part of the rule that is not in
that guarantee, and the field found the fixed cadence wrong on exactly the day
it mattered: when the cold lane first filled the maps on the alt store, the
curtailed share of a quiet day jumped 0.024 → 0.562 — the streaks strangled
the channel at the moment it first had something to say
(``~/p/ae/artifacts/injection-throttle/prereg-draft.md``, П2). So behind
:data:`CURTAIL_DECAY_GATE_ENV` the retry decays instead of waiting for the
window: after ``N`` unaccepted offers the key spends
``min(2**(N-1), CURTAIL_DECAY_SKIP_MAX)`` deliveries on markers and then
offers a full map again, and the first consumed delivery resets the walk
exactly as it always did. The entry threshold, the markers, the probes and
the evidence window are byte-identical either way; the valve can shorten a
silence and can never start one, and with it unset the cadence is exactly the
window arithmetic above.

The first field read of that valve (``~/p/ae/artifacts/injection-throttle/
read-2026-09-02/READ.md``) found the retry firing into nothing: a served skip
run rebuilt the map over a pool that had nothing deliverable in it, the empty
map counted as one more marker-shaped delivery, and the next build retried
again — every delivery after the run became an *empty, uncollapsed* map, so
the curtailed share fell to nothing while the empty share stayed at the storm
level. So the retry is pool-aware. The first marker after an offer persists a
digest of the key's pool (``pd``: the admitted members, the anchors that would
partition them, the cold-lane candidates) inside the marker it was going to
write anyway. A served skip run rebuilds the pool and compares: unchanged, the
key writes the same digest into one more marker and the run starts over —
not an offer, the streak's offer count does not move; changed, it builds the
full map, and if that map has nothing deliverable in it either, the marker
carries the new digest instead. The decay wakes only when the key has
something new to say. The digest lives in the same window the streak is read
from, so it survives a restart and needs no table; with the valve unset it is
neither computed nor written.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from math import fsum, isfinite, log, log1p, sqrt
from typing import TYPE_CHECKING, Any
import hashlib
import json
import os
import re
import sqlite3

from living_memory.embeddings import cosine_similarity, tokenize
from living_memory.grounding import token_set
from living_memory.models import Node
from living_memory.scope import GLOBAL_SCOPE, normalize_scope
from living_memory.storage import (
    CHUNK_EMBEDDING_TABLE,
    MAX_RECALL_HISTORY_CANDIDATES,
    RECALL_DELIVERY_HISTORY_STATE_TABLE,
    RECALL_DELIVERY_HISTORY_TABLE,
    MemoryStore,
    _RECALL_DELIVERY_HISTORY_FORMAT,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from living_memory.retrieval import RecallResult


# ----------------------------------------------------------------------
# Caps. All of them are module constants because two sibling delivery
# surfaces (the recall response and the server instructions) budget against
# them and must not each invent their own number.
# ----------------------------------------------------------------------

#: Most clusters a map may carry. Six is the point past which a "plan" stops
#: being one: the agent has to choose what to ask next, and a list longer than
#: this is re-read as a wall rather than as a menu.
MAX_CLUSTERS = 6

#: Budget for :meth:`RecallMap.to_dict` serialized compactly. The recall
#: response is on a delivery diet; the map is additive and must stay a
#: rounding error next to the results it decorates.
#:
#: The two caps bind in different regimes and it is worth knowing which. One
#: serialized cluster costs 88 characters of JSON keys before a single
#: character of content, so six of them spend 528 of this budget on
#: punctuation: :data:`MAX_CLUSTERS` is the ceiling for short labels, and for
#: anything richer *this* is the binding constraint. Examples narrow to their
#: informative floor, then tail clusters give way down to the breadth floor,
#: then journal names yield before gist does (:meth:`RecallMapBuilder._fit`). A
#: dropped cluster is counted in the payload's ``more`` field rather than
#: vanishing, because a map that silently truncates reads as "this is
#: everything memory holds", which is the one thing it must never say.
MAX_RESPONSE_CHARS = 700

#: Budget for :meth:`RecallMap.render_compact`, the single-string form the
#: server-instructions channel embeds. Instructions are capped at 2048 chars
#: by contract test and currently run within single digits of that ceiling, so
#: the map's whole footprint there is this line.
MAX_INSTRUCTIONS_CHARS = 150

#: Longest medoid example a map carries before the response budget starts
#: taking it apart. See :meth:`RecallMapBuilder._fit`.
MEDOID_EXAMPLE_CHARS = 120

#: Shortest informative gist the response budget should buy for every medoid.
#:
#: This is a floor on what the source text can supply: a ten-character memory
#: contributes all ten characters and satisfies the rule.  On the measured
#: field corpus every medoid was longer than the floor, so the ordinary case
#: is exactly forty or more characters in every delivered row.
MIN_MEDOID_EXAMPLE_CHARS = 40

#: Breadth that gist fitting may not cross.  Two clusters preserved 64 of 67
#: historically followed deliveries; one preserved only 57.  The value is
#: clamped to the number of clusters the configured builder actually kept, so
#: a one-cluster pool (or ``max_clusters=1``) still yields a map.
MIN_CLUSTERS = 2

MAX_LABEL_CHARS = 40
MAX_ASK_HINT_CHARS = 80

#: Clustering cap, applied only after the complete residual has passed frozen
#: eligibility and relevance scoring.  That ordering is load-bearing: applying
#: the cap while scanning would let a machine-ballast head crowd useful members
#: out before the selector saw them.  Only the admitted 200 pay for the heavier
#: four-stage cascade and its per-node anchor/vector reads.
MAX_POOL_NODES = 200

#: Frozen member-selection policy.  These are literals rather than a runtime
#: read of ``artifacts/``: a deployed server must not depend on its source
#: checkout being present, and changing any value is a policy revision rather
#: than configuration.  The digest binds the canonical ``selected_policy``
#: object in ``artifacts/recall-map/relevance/policy.json``.
RELEVANCE_POLICY_ID = "directional-zsum-r1"
RELEVANCE_POLICY_DIGEST = (
    "3acad3d92db2538bf3096ab99d2c4d337ea3c8aebddc226646527b4d277660ea"
)
RELEVANCE_FEATURE_MEANS: tuple[float, ...] = (
    0.26434558349451964,
    2.856678070667311,
    2.789348366105738,
    1.5799800264635717,
    0.06059739660863959,
)
RELEVANCE_FEATURE_SCALES: tuple[float, ...] = (
    0.4409841221421261,
    2.5772848149641323,
    2.5560016163553305,
    1.765967625291225,
    0.11674776461860277,
)
RELEVANCE_THRESHOLD = 3.8708378402511

#: The calibration tables the selected policy publishes, keyed by the
#: calibrated quantity.  ``directional-zsum-r1`` publishes none and this is
#: empty on purpose: it z-scores five features against
#: :data:`RELEVANCE_FEATURE_MEANS` and :data:`RELEVANCE_FEATURE_SCALES` and
#: calibrates nothing.  Every lookup against an empty block raises, which is
#: what keeps the reader below inert until a revision fills this in — and
#: filling it in is a policy revision that moves
#: :data:`RELEVANCE_POLICY_ID` and :data:`RELEVANCE_POLICY_DIGEST` together.
RELEVANCE_POLICY_CALIBRATION: Mapping[str, Any] = {}

#: Ceiling on the breakpoints one published table may spend, from
#: ``coldstart-prereg-v2.json#scorer_shape.calibration``
#: ``.out_of_sample_and_live_application.max_breakpoints``.  A 512-knot
#: quantile grid plus its atoms has to fit under it; a table that does not is
#: not the registered form and is refused rather than truncated.
RELEVANCE_CALIBRATION_MAX_BREAKPOINTS = 1024

#: ``max_interpolation_error_on_u`` from the same registered form: 2**-9, the
#: resolution a 512-knot grid buys.  Between two knots the reconstructed ``u``
#: and the true one both lie inside the knots' own ``u`` bracket, so holding
#: consecutive knots to this step is what makes the published bound true.
#:
#: These two are the registered *form*, not fitted content — the shape a table
#: must have to be read at all.  Not one number a lookup can return lives in
#: this module; every one of them comes off the table the policy publishes.
RELEVANCE_CALIBRATION_MAX_INTERPOLATION_ERROR = 0.001953125

#: Compact exclusion vector order frozen in the additive ``sel`` contract.
#: Do not alphabetize it: both persistence and the evaluator interpret counts
#: positionally.
SELECTION_REASON_CODES: tuple[str, ...] = (
    "iv",
    "du",
    "fc",
    "ss",
    "sj",
    "lr",
    "pc",
)
#: Reason codes the two env-gated pool gates add *after* the frozen vector.
#:
#: Appended, never interleaved: a reader that knows the seven above keeps
#: reading them at the same indices, which is the whole content of the
#: positional contract. The other half of keeping that promise is
#: :meth:`_SelectionLedger.freeze`, which trims trailing zeros back to the
#: frozen width — so a server with both valves unset emits exactly the
#: seven-wide ``x`` it emitted before these codes existed, and a longer vector
#: is itself the statement that a gate fired.
POOL_GATE_REASON_CODES: tuple[str, ...] = (
    #: Below the usefulness floor of :data:`POOL_USEFULNESS_FLOOR_ENV`.
    "uf",
    #: Demoted after :data:`POOL_DEMOTION_WINDOWS_ENV` known delivery windows
    #: that nobody followed.
    "nf",
)
SELECTION_LEDGER_REASON_CODES: tuple[str, ...] = (
    *SELECTION_REASON_CODES,
    *POOL_GATE_REASON_CODES,
)
SELECTION_SAMPLE_GIST_CHARS = 24
SELECTION_SAMPLE_LIMIT = 2
#: Both gate codes are sampleable, and that is a decision rather than a
#: default. The three ballast codes are sampleable because an operator reading
#: the journal needs to see *what* a machine-shaped rule swallowed; ``lr`` and
#: ``pc`` because a frozen score and a cap are invisible otherwise. The gate
#: codes are the strongest case of the same argument: their thresholds are the
#: one part of this module an operator sets by hand from field numbers, and a
#: journal that says "the floor dropped 14" without ever naming one of them
#: gives the operator no way to tell a floor that is working from a floor set
#: one decimal place too high. The gist is already identity-free and capped at
#: :data:`SELECTION_SAMPLE_GIST_CHARS`, so admitting them costs the payload
#: nothing it does not already spend on the codes beside them — and costs it
#: only when a valve is on, since with both unset neither code ever fires.
_SAMPLEABLE_REASONS = frozenset({"fc", "ss", "sj", "lr", "pc", "uf", "nf"})

# ----------------------------------------------------------------------
# Pool gates behind env valves
# ----------------------------------------------------------------------
#
# Two admission rules that ``_pool`` applies *around* the frozen selector,
# never inside it: :data:`RELEVANCE_FEATURE_MEANS`,
# :data:`RELEVANCE_FEATURE_SCALES` and :func:`relevance_score` see exactly what
# they saw before, and every new signal enters as a separate yes/no on a
# candidate the frozen pipeline already judged.
#
# Both follow ``retrieval.DRAIN_NEAR_DUP_ENV`` and take it one step further.
# There the flag is a whole env var — rather than a threshold of zero — because
# turning the behaviour on has to be a sentence an operator said. Here the
# *number* is a second env var with no shipped default at all, because unlike
# the drain's cosine there is no measured value to ship: the field distribution
# these thresholds come from is a sibling goal's artifact, and the operator
# reads them off it. A valve that is on with no number is inert, which is the
# only safe reading of "on, but at what?".

#: ``1``/``true``/``yes``/``on``, exactly as the drain valve parses. Anything
#: else — unset, empty, ``0``, a typo — is off, because a misread flag here
#: silently removes rows from the one channel that says what memory holds.
_POOL_GATE_ON_FLAGS = frozenset({"1", "true", "yes", "on"})

#: Valve for gate (a): drop a candidate whose ``usefulness_score`` sits below
#: the floor. The score is already on the node the residual carries, so this
#: gate costs one attribute read per candidate and no query at all.
POOL_USEFULNESS_GATE_ENV = "LM_MAP_POOL_USEFULNESS_GATE"
#: The floor itself. No default: see the block comment above.
POOL_USEFULNESS_FLOOR_ENV = "LM_MAP_POOL_MIN_USEFULNESS"

#: Valve for gate (b): demote a row that has been delivered into N known
#: windows in a row without anybody following it.
POOL_DEMOTION_GATE_ENV = "LM_MAP_POOL_DEMOTION_GATE"
#: N itself, in windows. No default: see the block comment above.
POOL_DEMOTION_WINDOWS_ENV = "LM_MAP_POOL_DEMOTE_AFTER"

#: What one re-delivered window is worth against the demotion run.
#:
#: Re-delivery — "a later recall returned this node again" — is the ranker's
#: own echo: the schema-trigger boost mixes hub schemas into nearly every
#: recall, so a hub carries ``consumed == matured`` forever without a reader
#: ever opening it. It is still evidence of *something*, so it softens; it may
#: never veto, because a veto is precisely what makes those hubs immortal and
#: is the failure this gate exists to fix.
#:
#: Hence a weight strictly inside ``(0, 1)``: ``0`` is no softening at all and
#: ``1`` is the veto. Which value inside the interval is not a threshold to
#: tune here — the run is compared against the operator's N, so any fixed
#: weight is absorbed by their choice of N. At ``0.5`` the arithmetic states
#: itself: a row nothing ever re-delivered sinks after N windows, a row
#: re-delivered every single time needs 2N, and every mixture lands in
#: between.
DEMOTION_REDELIVERY_WEIGHT = 0.5

# ----------------------------------------------------------------------
# Cold exploration lane behind its own paired env valve
# ----------------------------------------------------------------------
#
# Registered in ``artifacts/recall-map/pool-quality/cold-quota-prereg.json``
# (plan cold-quota-r1). The lane never touches the frozen selector: candidates
# the frozen threshold alone excluded (``lr``) and that the delivery ledger has
# never seen are ranked only against each other on their own candidate-time
# scores, and at most :data:`POOL_COLD_SLOTS_MAX` of them are appended to the
# *built* map as marked singleton clusters, into free cluster capacity only.
# With the valve pair unset nothing below runs and the payload is byte-for-byte
# what the pre-lane module emitted.

#: Valve for the cold quota lane, drain-valve spelling like the two gates
#: above. On with no parseable slot count is inert: "on, but at what?" has no
#: safe answer.
POOL_COLD_QUOTA_GATE_ENV = "LM_MAP_POOL_COLD_QUOTA_GATE"
#: The slot count itself. The registered domain is ``1..POOL_COLD_SLOTS_MAX``;
#: anything else — non-integer, empty, ``0``, negative, larger — leaves the
#: lane inert, exactly as :func:`pool_demotion_windows_from_env` rejects ``0``.
POOL_COLD_SLOTS_ENV = "LM_MAP_POOL_COLD_SLOTS"
#: The registered quota is a 1-2 slot exploration lane; a larger number is not
#: this design and reads as the misconfiguration it is.
POOL_COLD_SLOTS_MAX = 2

#: The four registered composite members and their signs, family
#: ``recorded-delivery-scores-v3``: ``f = u_rs + u_tg + (1-u_bm) + (1-u_vs)``.
#: ``graph_score`` is excluded permanently (its association flips sign on the
#: cold subpopulation); ``usefulness_score`` and ``access_count`` are banned as
#: mutable post-outcome stats.
COLD_RANKING_MEMBERS: tuple[tuple[str, float], ...] = (
    ("result_score", 1.0),
    ("trigger_score", 1.0),
    ("bm25_score", -1.0),
    ("vector_score", -1.0),
)

#: Registered member name -> the field carrying it on the residual result
#: record, the same per-item fields ``recall_events.results`` persists.
_COLD_MEMBER_FIELDS: Mapping[str, str] = {
    "result_score": "score",
    "trigger_score": "trigger_score",
    "bm25_score": "bm25_score",
    "vector_score": "vector_score",
}

# --- BEGIN COLD_RANKING_CALIBRATION (generated) -----------------------
#: Per-member monotone rank tables fitted on the cold-eligible subpopulation
#: of the replayed live candidate population (500 queries, seed 20260824,
#: frozen snapshot c70875d9...), in the registered form
#: :func:`_calibration_table` reads: atoms at >= 0.5% multiplicity, a quantile
#: grid of at most 512 knots, at most 1024 breakpoints, u-steps within
#: :data:`RELEVANCE_CALIBRATION_MAX_INTERPOLATION_ERROR`. Digests and the
#: fitting-population count are published in
#: ``artifacts/recall-map/pool-quality/cold-quota-params.json``; regenerated
#: only by ``scripts/recall_map_cold_lane_replay.py fit``. The frozen warm
#: policy (:data:`RELEVANCE_POLICY_CALIBRATION`) stays empty and does not
#: move.
COLD_RANKING_CALIBRATION: Mapping[str, Any] = {
    "bm25_score": {
        "atoms": [
            [
                0.0,
                0.472600538
            ]
        ],
        "knots": [
            [
                0.025,
                0.945856521
            ],
            [
                0.02564102564102564,
                0.947284003
            ],
            [
                0.02631578947368421,
                0.948742997
            ],
            [
                0.02702702702702703,
                0.95009485
            ],
            [
                0.027777777777777776,
                0.951475065
            ],
            [
                0.02857142857142857,
                0.952889942
            ],
            [
                0.029411764705882353,
                0.954292215
            ],
            [
                0.030303030303030304,
                0.955719697
            ],
            [
                0.03125,
                0.957128272
            ],
            [
                0.03225806451612903,
                0.958562056
            ],
            [
                0.03333333333333333,
                0.960046259
            ],
            [
                0.034482758620689655,
                0.96147059
            ],
            [
                0.03571428571428571,
                0.962828746
            ],
            [
                0.037037037037037035,
                0.964224717
            ],
            [
                0.038461538461538464,
                0.965674257
            ],
            [
                0.04,
                0.967085983
            ],
            [
                0.041666666666666664,
                0.968456744
            ],
            [
                0.043478260869565216,
                0.9698149
            ],
            [
                0.045454545454545456,
                0.971207719
            ],
            [
                0.047619047619047616,
                0.972622596
            ],
            [
                0.05,
                0.974072137
            ],
            [
                0.05263157894736842,
                0.975474409
            ],
            [
                0.05555555555555555,
                0.97681681
            ],
            [
                0.058823529411764705,
                0.978187571
            ],
            [
                0.0625,
                0.979492157
            ],
            [
                0.06666666666666667,
                0.98081565
            ],
            [
                0.07142857142857142,
                0.982195864
            ],
            [
                0.07692307692307693,
                0.983569776
            ],
            [
                0.08333333333333333,
                0.98494684
            ],
            [
                0.09090909090909091,
                0.98628924
            ],
            [
                0.1,
                0.987666303
            ],
            [
                0.1111111111111111,
                0.989131599
            ],
            [
                0.125,
                0.990546477
            ],
            [
                0.14285714285714285,
                0.991892028
            ],
            [
                0.16666666666666666,
                0.993256487
            ],
            [
                0.2,
                0.994592585
            ],
            [
                0.25,
                0.995938136
            ],
            [
                0.3333333333333333,
                0.997255327
            ],
            [
                0.5,
                0.998421262
            ],
            [
                1.0,
                0.999483207
            ]
        ]
    },
    "result_score": {
        "atoms": [
            [
                0.039,
                0.741379962
            ],
            [
                0.051333750000000004,
                0.781043795
            ]
        ],
        "knots": [
            [
                4.282968937171783e-05,
                3.151e-06
            ],
            [
                0.0009461752177881467,
                0.001947426
            ],
            [
                0.001476141222381592,
                0.003888549
            ],
            [
                0.0018205792366504667,
                0.005839126
            ],
            [
                0.0019856783282361764,
                0.007786552
            ],
            [
                0.0021046888413906095,
                0.009730827
            ],
            [
                0.002186483554601669,
                0.01159002
            ],
            [
                0.0022364569677114487,
                0.013534294
            ],
            [
                0.002296624041080475,
                0.015465964
            ],
            [
                0.0023672369458200187,
                0.017410239
            ],
            [
                0.002453377796302708,
                0.019357665
            ],
            [
                0.002533386990460754,
                0.021220009
            ],
            [
                0.0025749635520451375,
                0.023151679
            ],
            [
                0.0026392807989499595,
                0.02500772
            ],
            [
                0.0027058621883638313,
                0.026863762
            ],
            [
                0.0027809506453428854,
                0.028814339
            ],
            [
                0.0028359841226041323,
                0.030755463
            ],
            [
                0.002886807134650684,
                0.032687132
            ],
            [
                0.002957662327672812,
                0.034628256
            ],
            [
                0.003052756327954761,
                0.03654417
            ],
            [
                0.003128278491318226,
                0.038472689
            ],
            [
                0.0032046386248739487,
                0.040420115
            ],
            [
                0.0032797253319025044,
                0.042351785
            ],
            [
                0.003350973273731074,
                0.04429921
            ],
            [
                0.003436963447070567,
                0.046246636
            ],
            [
                0.0034890716460049156,
                0.048190911
            ],
            [
                0.003582290194094181,
                0.050135185
            ],
            [
                0.0036504000000000003,
                0.051975471
            ],
            [
                0.0037341027748643376,
                0.053919746
            ],
            [
                0.0038231917073307314,
                0.055867172
            ],
            [
                0.003906475886107377,
                0.05776733
            ],
            [
                0.003982310922369361,
                0.059705302
            ],
            [
                0.004057841375693679,
                0.061652728
            ],
            [
                0.004134016616106034,
                0.063593851
            ],
            [
                0.00422309677746892,
                0.065525521
            ],
            [
                0.0043247791065818076,
                0.067476098
            ],
            [
                0.0044178200245181655,
                0.069423524
            ],
            [
                0.0045618652026319685,
                0.071197635
            ],
            [
                0.004619852059967816,
                0.07303477
            ],
            [
                0.004663119691626597,
                0.074972742
            ],
            [
                0.004784651707511844,
                0.076910715
            ],
            [
                0.0048963957887591305,
                0.078854989
            ],
            [
                0.005024140456858695,
                0.080792962
            ],
            [
                0.005116006341040135,
                0.082740387
            ],
            [
                0.005216351612091064,
                0.084684662
            ],
            [
                0.00533935211404589,
                0.086635239
            ],
            [
                0.005449455709218979,
                0.088585816
            ],
            [
                0.005544045000000001,
                0.090504881
            ],
            [
                0.0056546893383264555,
                0.092427098
            ],
            [
                0.005764895067662,
                0.09436507
            ],
            [
                0.005868560269474983,
                0.096293589
            ],
            [
                0.005925881534814835,
                0.098218956
            ],
            [
                0.005973523898643887,
                0.100166382
            ],
            [
                0.006024103224277496,
                0.102110657
            ],
            [
                0.006079442836493252,
                0.104061234
            ],
            [
                0.006137898445129395,
                0.105967694
            ],
            [
                0.006186870396137237,
                0.107877306
            ],
            [
                0.006216876772999764,
                0.109821581
            ],
            [
                0.006273001939058304,
                0.111731192
            ],
            [
                0.006332211068292246,
                0.113672316
            ],
            [
                0.00637757760827642,
                0.115613439
            ],
            [
                0.006409764021635055,
                0.117513597
            ],
            [
                0.0064480405747890475,
                0.119410604
            ],
            [
                0.006499627758345451,
                0.121351728
            ],
            [
                0.006554412774654947,
                0.123302305
            ],
            [
                0.006606245919168448,
                0.12522137
            ],
            [
                0.006646219354348941,
                0.127159342
            ],
            [
                0.006702675449538279,
                0.129109919
            ],
            [
                0.006747687722241232,
                0.131060496
            ],
            [
                0.006803209964454175,
                0.132995317
            ],
            [
                0.006844234049320221,
                0.134403892
            ],
            [
                0.0068587779566807875,
                0.136348167
            ],
            [
                0.006903878342986108,
                0.138295593
            ],
            [
                0.00694894347332418,
                0.140236716
            ],
            [
                0.006993079885840416,
                0.142171537
            ],
            [
                0.007046084799170495,
                0.144093754
            ],
            [
                0.007092138095199943,
                0.146038028
            ],
            [
                0.007141609096380384,
                0.147985454
            ],
            [
                0.007192219074967256,
                0.149923426
            ],
            [
                0.007267094489859165,
                0.151874003
            ],
            [
                0.007315850593149662,
                0.153767859
            ],
            [
                0.007369902112142022,
                0.155708983
            ],
            [
                0.007415421847254038,
                0.157643804
            ],
            [
                0.007463998818062265,
                0.159594381
            ],
            [
                0.0075152413332613,
                0.161494539
            ],
            [
                0.007568369621481563,
                0.163363185
            ],
            [
                0.007614129780828952,
                0.165310611
            ],
            [
                0.007653532198369503,
                0.167245432
            ],
            [
                0.0077006516000032435,
                0.169186556
            ],
            [
                0.007761357272507116,
                0.171118226
            ],
            [
                0.007797557567593997,
                0.173065652
            ],
            [
                0.007845800292491913,
                0.174978414
            ],
            [
                0.007878600286338529,
                0.176922689
            ],
            [
                0.007928509205357526,
                0.178870115
            ],
            [
                0.00797464511203766,
                0.180820692
            ],
            [
                0.008034071280956267,
                0.182771269
            ],
            [
                0.008082258193597197,
                0.184715543
            ],
            [
                0.008144693478763104,
                0.18666612
            ],
            [
                0.008203045384347438,
                0.188607244
            ],
            [
                0.008254227730751037,
                0.190551519
            ],
            [
                0.008318088782072068,
                0.192489491
            ],
            [
                0.00837094301590696,
                0.194436917
            ],
            [
                0.00842077005493641,
                0.196292958
            ],
            [
                0.008447621428906918,
                0.198199419
            ],
            [
                0.008495497635126114,
                0.200146845
            ],
            [
                0.008555292561650276,
                0.201810665
            ],
            [
                0.00859793494552374,
                0.203758091
            ],
            [
                0.008649024276137354,
                0.205702365
            ],
            [
                0.008712378311902285,
                0.207643489
            ],
            [
                0.008757394319529453,
                0.209584612
            ],
            [
                0.008798389684438706,
                0.211535189
            ],
            [
                0.008846808839589358,
                0.213482615
            ],
            [
                0.008940586764433737,
                0.215430041
            ],
            [
                0.009030240516185761,
                0.217371164
            ],
            [
                0.009099324162499998,
                0.21931859
            ],
            [
                0.009179460629852308,
                0.221266016
            ],
            [
                0.009270689597010613,
                0.22321029
            ],
            [
                0.00935854949662796,
                0.224927681
            ],
            [
                0.009429656411309204,
                0.226849897
            ],
            [
                0.009520811504065991,
                0.228787869
            ],
            [
                0.009610041592195633,
                0.230738446
            ],
            [
                0.009691663262122337,
                0.232685872
            ],
            [
                0.009756540951468051,
                0.234614391
            ],
            [
                0.009857855756789444,
                0.236561817
            ],
            [
                0.009962493225129992,
                0.238496638
            ],
            [
                0.010060228044959795,
                0.240437761
            ],
            [
                0.010190631987493773,
                0.242378885
            ],
            [
                0.010314854462072252,
                0.244320008
            ],
            [
                0.010400583655008771,
                0.245977526
            ],
            [
                0.010464784594528377,
                0.247924952
            ],
            [
                0.010582799223953377,
                0.249875529
            ],
            [
                0.010694774214715248,
                0.251822954
            ],
            [
                0.010817956892271212,
                0.253767229
            ],
            [
                0.01093304598614845,
                0.255708353
            ],
            [
                0.011034821919654214,
                0.257655778
            ],
            [
                0.011140710512135358,
                0.259593751
            ],
            [
                0.011262006671394096,
                0.261538025
            ],
            [
                0.011346733868122101,
                0.263353102
            ],
            [
                0.01144816925033927,
                0.26522805
            ],
            [
                0.011536523031815887,
                0.267166023
            ],
            [
                0.011635357408225539,
                0.269047274
            ],
            [
                0.011735834686068558,
                0.270991549
            ],
            [
                0.011844385714396838,
                0.272932672
            ],
            [
                0.011909416895359756,
                0.274883249
            ],
            [
                0.012000934365590897,
                0.276830675
            ],
            [
                0.01210829738096174,
                0.278781252
            ],
            [
                0.012185971750994774,
                0.280659352
            ],
            [
                0.012261952701956035,
                0.282600475
            ],
            [
                0.01237814560621977,
                0.284494331
            ],
            [
                0.012482406770586775,
                0.286432303
            ],
            [
                0.012595606070045255,
                0.288379729
            ],
            [
                0.012692136276067708,
                0.290327155
            ],
            [
                0.01282054100109074,
                0.292265127
            ],
            [
                0.012861275422748758,
                0.294215704
            ],
            [
                0.012969581541921796,
                0.296141072
            ],
            [
                0.013067653253674509,
                0.298075893
            ],
            [
                0.013178441938273612,
                0.30002647
            ],
            [
                0.013310920595079663,
                0.301967593
            ],
            [
                0.013447142831459643,
                0.303908717
            ],
            [
                0.013552184889949859,
                0.305856143
            ],
            [
                0.013665968361496925,
                0.307746847
            ],
            [
                0.01368867036509514,
                0.308874968
            ],
            [
                0.01368883518254757,
                0.309988908
            ],
            [
                0.013689000000000003,
                0.311102848
            ],
            [
                0.013689000815927985,
                0.312216788
            ],
            [
                0.013689001631855964,
                0.313330728
            ],
            [
                0.013814735725969078,
                0.315271852
            ],
            [
                0.01393925653517246,
                0.317165708
            ],
            [
                0.01403526669368148,
                0.319094226
            ],
            [
                0.014143131030008704,
                0.321044803
            ],
            [
                0.014268686438128354,
                0.322989078
            ],
            [
                0.014346525893992577,
                0.324876632
            ],
            [
                0.014455424511117561,
                0.326811453
            ],
            [
                0.014583370482623582,
                0.328717913
            ],
            [
                0.014695798729173839,
                0.330599164
            ],
            [
                0.014767680964507165,
                0.332549741
            ],
            [
                0.014870844348818066,
                0.334497167
            ],
            [
                0.014997553884834052,
                0.336441442
            ],
            [
                0.015086948484443129,
                0.338351053
            ],
            [
                0.015176365138031542,
                0.340298479
            ],
            [
                0.015301781302280727,
                0.342249056
            ],
            [
                0.015425284829735757,
                0.34419018
            ],
            [
                0.0155296137907058,
                0.346087187
            ],
            [
                0.015626646906137468,
                0.348034613
            ],
            [
                0.01572789765123278,
                0.349982038
            ],
            [
                0.015821723751962273,
                0.351932615
            ],
            [
                0.015927374969619514,
                0.353870588
            ],
            [
                0.01603569424364716,
                0.355792804
            ],
            [
                0.01613640184164047,
                0.357737079
            ],
            [
                0.016275209902860226,
                0.359668749
            ],
            [
                0.016394296423305713,
                0.361613023
            ],
            [
                0.01651026042431887,
                0.363560449
            ],
            [
                0.016661740995794535,
                0.365476363
            ],
            [
                0.016836968904733657,
                0.367423789
            ],
            [
                0.016984883663691584,
                0.369273528
            ],
            [
                0.017106994509869883,
                0.371167384
            ],
            [
                0.017130510427000673,
                0.37311481
            ],
            [
                0.017272127111823857,
                0.375059085
            ],
            [
                0.017450006979703902,
                0.377009662
            ],
            [
                0.017565531224012377,
                0.378941331
            ],
            [
                0.01773470377922058,
                0.380879304
            ],
            [
                0.017888491094112398,
                0.382823578
            ],
            [
                0.018011815878136547,
                0.384774155
            ],
            [
                0.018154807080658454,
                0.386724732
            ],
            [
                0.018295217382396742,
                0.388665856
            ],
            [
                0.018466181520620652,
                0.390600677
            ],
            [
                0.018645063404459,
                0.392551254
            ],
            [
                0.018856589956366396,
                0.394470319
            ],
            [
                0.018973475698828696,
                0.396370477
            ],
            [
                0.019095763474702838,
                0.398314752
            ],
            [
                0.019198217064142226,
                0.400218061
            ],
            [
                0.019302414089441302,
                0.402168638
            ],
            [
                0.019409266948699953,
                0.40407825
            ],
            [
                0.01949906464666128,
                0.405997315
            ],
            [
                0.019561318545341492,
                0.407947892
            ],
            [
                0.019696051252999263,
                0.409750364
            ],
            [
                0.019804207454815512,
                0.411615859
            ],
            [
                0.019850206312738358,
                0.413566436
            ],
            [
                0.019976350843906403,
                0.415479199
            ],
            [
                0.02011906152963638,
                0.417423474
            ],
            [
                0.020258055210113524,
                0.4193709
            ],
            [
                0.020385786294937133,
                0.421308872
            ],
            [
                0.020486126840114593,
                0.423237391
            ],
            [
                0.02058765059709549,
                0.425187968
            ],
            [
                0.020696264326572417,
                0.427135393
            ],
            [
                0.02081342484610677,
                0.429054459
            ],
            [
                0.02091656720638275,
                0.430995582
            ],
            [
                0.021059774874486032,
                0.432943008
            ],
            [
                0.021156087398529054,
                0.434871527
            ],
            [
                0.02125491081686318,
                0.436806348
            ],
            [
                0.021355343520641328,
                0.43869075
            ],
            [
                0.02143974551296234,
                0.440531036
            ],
            [
                0.021528930884879473,
                0.442472159
            ],
            [
                0.021646897024402426,
                0.444413283
            ],
            [
                0.02177008134126663,
                0.44636386
            ],
            [
                0.021826374175071713,
                0.448308134
            ],
            [
                0.021934793853600438,
                0.45025556
            ],
            [
                0.0220310621740669,
                0.452174625
            ],
            [
                0.02210688917167222,
                0.454106295
            ],
            [
                0.022206995666027067,
                0.456019058
            ],
            [
                0.022245697617530823,
                0.457944426
            ],
            [
                0.022310355842113494,
                0.459891852
            ],
            [
                0.02239812551710612,
                0.461839277
            ],
            [
                0.0224737948179245,
                0.463711075
            ],
            [
                0.02258031177520752,
                0.465661652
            ],
            [
                0.022659989485162495,
                0.467602776
            ],
            [
                0.022731335163116456,
                0.469537597
            ],
            [
                0.022810442984104156,
                0.471469267
            ],
            [
                0.022901901014149188,
                0.473394634
            ],
            [
                0.02299510051980363,
                0.475326304
            ],
            [
                0.023069877359715836,
                0.477248521
            ],
            [
                0.02315634123980999,
                0.479139225
            ],
            [
                0.023224882534203753,
                0.481055139
            ],
            [
                0.023292380152165892,
                0.48298996
            ],
            [
                0.02335870281443731,
                0.484902723
            ],
            [
                0.02348273487215639,
                0.486846998
            ],
            [
                0.02357147056232393,
                0.488797575
            ],
            [
                0.02366918868439079,
                0.490489756
            ],
            [
                0.02374627286195755,
                0.492437181
            ],
            [
                0.02379094237580939,
                0.494233351
            ],
            [
                0.02388487833738327,
                0.496183928
            ],
            [
                0.023958413265645503,
                0.498134505
            ],
            [
                0.02403465002061806,
                0.500050419
            ],
            [
                0.02413686264306307,
                0.501997845
            ],
            [
                0.024235118329524994,
                0.50394527
            ],
            [
                0.024301289993003013,
                0.505851731
            ],
            [
                0.024374965230077508,
                0.507651052
            ],
            [
                0.02445445127785206,
                0.509529152
            ],
            [
                0.024542815436157793,
                0.511476577
            ],
            [
                0.024630603012281295,
                0.513354677
            ],
            [
                0.024705965872853994,
                0.515305254
            ],
            [
                0.024777830487281084,
                0.517249529
            ],
            [
                0.024808868251740935,
                0.518913349
            ],
            [
                0.02483612483739853,
                0.520854472
            ],
            [
                0.02491090428829193,
                0.522748328
            ],
            [
                0.02501555181887001,
                0.524692603
            ],
            [
                0.02509262887048424,
                0.526640029
            ],
            [
                0.025170158442556857,
                0.528458256
            ],
            [
                0.025274824868202213,
                0.530408833
            ],
            [
                0.025360902808606627,
                0.53230584
            ],
            [
                0.025445744854438027,
                0.534240661
            ],
            [
                0.025532858828902247,
                0.536188087
            ],
            [
                0.02559676207602024,
                0.538138664
            ],
            [
                0.02566642671277004,
                0.539717403
            ],
            [
                0.02566665085638502,
                0.540738384
            ],
            [
                0.025666875000000002,
                0.541759364
            ],
            [
                0.025667582509517672,
                0.542785071
            ],
            [
                0.02566829001903534,
                0.543810778
            ],
            [
                0.02574880203232169,
                0.545745599
            ],
            [
                0.0258823681075573,
                0.547696176
            ],
            [
                0.026000867988914253,
                0.549643602
            ],
            [
                0.02612087424845123,
                0.551591028
            ],
            [
                0.02621194446958602,
                0.553525849
            ],
            [
                0.02627581327222288,
                0.555448065
            ],
            [
                0.02637447685003281,
                0.557386038
            ],
            [
                0.026462962283276166,
                0.559330312
            ],
            [
                0.026541573554277418,
                0.561227319
            ],
            [
                0.02666761077144026,
                0.563155838
            ],
            [
                0.026691431687772275,
                0.564863775
            ],
            [
                0.0267814795397222,
                0.566785991
            ],
            [
                0.026860995801240208,
                0.568670393
            ],
            [
                0.026947086726240226,
                0.570334214
            ],
            [
                0.02697100505232811,
                0.572168197
            ],
            [
                0.027068356758579614,
                0.574093565
            ],
            [
                0.027169599831104282,
                0.575943304
            ],
            [
                0.027255219665805887,
                0.577887579
            ],
            [
                0.027366937444433578,
                0.57979719
            ],
            [
                0.027396138980984686,
                0.581735163
            ],
            [
                0.027427125956833363,
                0.583654228
            ],
            [
                0.027512984052300456,
                0.585601654
            ],
            [
                0.027605740741506693,
                0.587514417
            ],
            [
                0.027688544025912882,
                0.589121516
            ],
            [
                0.027761087630001215,
                0.59101222
            ],
            [
                0.027873524690642956,
                0.592921832
            ],
            [
                0.027951426045941984,
                0.594837746
            ],
            [
                0.02799389446623624,
                0.596778869
            ],
            [
                0.028079507137656216,
                0.598726295
            ],
            [
                0.028206256671210003,
                0.600661116
            ],
            [
                0.02827885323442519,
                0.602599089
            ],
            [
                0.02840106190770865,
                0.604537061
            ],
            [
                0.028470966483199595,
                0.606452975
            ],
            [
                0.028558185363247994,
                0.60810419
            ],
            [
                0.02863635242963966,
                0.60998229
            ],
            [
                0.02869236349605024,
                0.611923414
            ],
            [
                0.028785835185796027,
                0.613864537
            ],
            [
                0.028830124776586898,
                0.615654404
            ],
            [
                0.028919606847129766,
                0.61760183
            ],
            [
                0.028951498143598438,
                0.619546105
            ],
            [
                0.029094608724117278,
                0.621477775
            ],
            [
                0.029191995976120235,
                0.623425201
            ],
            [
                0.029266243421882377,
                0.625183556
            ],
            [
                0.029357637515813117,
                0.627080563
            ],
            [
                0.029467136070951826,
                0.628898791
            ],
            [
                0.029554509718939664,
                0.630786344
            ],
            [
                0.029647448607712987,
                0.632727468
            ],
            [
                0.02972056126244366,
                0.634624475
            ],
            [
                0.029833133344463955,
                0.636540389
            ],
            [
                0.029925692215213236,
                0.638487814
            ],
            [
                0.03005230920922543,
                0.640422636
            ],
            [
                0.030162470119850557,
                0.642370061
            ],
            [
                0.030270865929946305,
                0.644150475
            ],
            [
                0.030336803110018375,
                0.646006517
            ],
            [
                0.03043275967238183,
                0.647953942
            ],
            [
                0.030583463908998514,
                0.649901368
            ],
            [
                0.03070806583555629,
                0.651848794
            ],
            [
                0.030748560146391397,
                0.653462195
            ],
            [
                0.030822278219684963,
                0.655321388
            ],
            [
                0.030875247350254306,
                0.657268814
            ],
            [
                0.03094456338621676,
                0.659184728
            ],
            [
                0.031049479041248558,
                0.661135305
            ],
            [
                0.03119078045322897,
                0.66307958
            ],
            [
                0.03130296607589722,
                0.664976587
            ],
            [
                0.03141287483882905,
                0.666816873
            ],
            [
                0.03154076594799233,
                0.668622496
            ],
            [
                0.03161221175640821,
                0.670573073
            ],
            [
                0.03173100883081556,
                0.672454324
            ],
            [
                0.03185589286176442,
                0.674360784
            ],
            [
                0.031976067514503034,
                0.67630821
            ],
            [
                0.03208993351042271,
                0.678198915
            ],
            [
                0.032185149246230726,
                0.679950968
            ],
            [
                0.03228381941713393,
                0.68183537
            ],
            [
                0.03236086647659541,
                0.68376704
            ],
            [
                0.032481799242496494,
                0.685714466
            ],
            [
                0.03255194293928147,
                0.687488577
            ],
            [
                0.03271610198756634,
                0.689423398
            ],
            [
                0.03284112125028006,
                0.691370824
            ],
            [
                0.03298367756433786,
                0.693226866
            ],
            [
                0.03305086617395282,
                0.6949285
            ],
            [
                0.033138177124972905,
                0.696809751
            ],
            [
                0.03328759441919625,
                0.698741421
            ],
            [
                0.03352949100340399,
                0.700691998
            ],
            [
                0.033743159497882994,
                0.702639424
            ],
            [
                0.033909024297773836,
                0.704567942
            ],
            [
                0.034063609385669234,
                0.706383019
            ],
            [
                0.034192218025445933,
                0.708311538
            ],
            [
                0.034455324025707654,
                0.710252661
            ],
            [
                0.034557774409460625,
                0.711998412
            ],
            [
                0.034737761457759,
                0.713945838
            ],
            [
                0.03502963902334686,
                0.715893263
            ],
            [
                0.03538070223927497,
                0.717645316
            ],
            [
                0.03564049729663182,
                0.719592742
            ],
            [
                0.03602043980008364,
                0.721540168
            ],
            [
                0.03639383347705007,
                0.723490745
            ],
            [
                0.03668851259404421,
                0.725441322
            ],
            [
                0.036995181553065776,
                0.727379294
            ],
            [
                0.03731054263472557,
                0.729099836
            ],
            [
                0.03759467868964355,
                0.730905458
            ],
            [
                0.03784752475743375,
                0.732843431
            ],
            [
                0.03826455471217632,
                0.734794008
            ],
            [
                0.038672255577027796,
                0.736719375
            ],
            [
                2.9039866002240435,
                0.999996849
            ]
        ]
    },
    "trigger_score": {
        "atoms": [
            [
                0.0,
                0.499322497
            ]
        ],
        "knots": [
            [
                0.975,
                0.998802554
            ],
            [
                1.0,
                0.999873953
            ]
        ]
    },
    "vector_score": {
        "atoms": [
            [
                0.0,
                0.439758998
            ]
        ],
        "knots": [
            [
                0.18184927105903625,
                0.879521148
            ],
            [
                0.3434552848339081,
                0.881468573
            ],
            [
                0.36580488085746765,
                0.883415999
            ],
            [
                0.3810511827468872,
                0.885363425
            ],
            [
                0.39168107509613037,
                0.887310851
            ],
            [
                0.4004264771938324,
                0.889258277
            ],
            [
                0.4085971415042877,
                0.891205702
            ],
            [
                0.4161820709705353,
                0.893153128
            ],
            [
                0.42227402329444885,
                0.895100554
            ],
            [
                0.427592009305954,
                0.89704798
            ],
            [
                0.4330182671546936,
                0.898995406
            ],
            [
                0.4380373954772949,
                0.900942831
            ],
            [
                0.44239887595176697,
                0.902890257
            ],
            [
                0.44636088609695435,
                0.904837683
            ],
            [
                0.450370192527771,
                0.906785109
            ],
            [
                0.4546162188053131,
                0.908732535
            ],
            [
                0.4583626985549927,
                0.91067996
            ],
            [
                0.46221446990966797,
                0.912627386
            ],
            [
                0.4655419588088989,
                0.914574812
            ],
            [
                0.4692562222480774,
                0.916522238
            ],
            [
                0.4724314510822296,
                0.918469664
            ],
            [
                0.475703626871109,
                0.920417089
            ],
            [
                0.47869598865509033,
                0.922364515
            ],
            [
                0.4819644093513489,
                0.924311941
            ],
            [
                0.4851280152797699,
                0.926259367
            ],
            [
                0.48848050832748413,
                0.928206793
            ],
            [
                0.49138161540031433,
                0.930154218
            ],
            [
                0.4945862889289856,
                0.932101644
            ],
            [
                0.49773648381233215,
                0.93404907
            ],
            [
                0.500457763671875,
                0.935996496
            ],
            [
                0.5033261775970459,
                0.937943922
            ],
            [
                0.5061900615692139,
                0.939891348
            ],
            [
                0.5091306567192078,
                0.941838773
            ],
            [
                0.5117607116699219,
                0.943786199
            ],
            [
                0.514281153678894,
                0.945733625
            ],
            [
                0.5171493291854858,
                0.947681051
            ],
            [
                0.5197951793670654,
                0.949628477
            ],
            [
                0.5224427580833435,
                0.951575902
            ],
            [
                0.5253888964653015,
                0.953523328
            ],
            [
                0.5283223390579224,
                0.955470754
            ],
            [
                0.531343936920166,
                0.95741818
            ],
            [
                0.5338909029960632,
                0.959365606
            ],
            [
                0.5368397235870361,
                0.961313031
            ],
            [
                0.5395470261573792,
                0.963260457
            ],
            [
                0.5423921346664429,
                0.965211034
            ],
            [
                0.5454372763633728,
                0.967161611
            ],
            [
                0.5485271215438843,
                0.969109037
            ],
            [
                0.5518088340759277,
                0.971056463
            ],
            [
                0.5548797845840454,
                0.973003889
            ],
            [
                0.558305025100708,
                0.974951314
            ],
            [
                0.5623133778572083,
                0.97689874
            ],
            [
                0.566662609577179,
                0.978846166
            ],
            [
                0.5710911154747009,
                0.980793592
            ],
            [
                0.5755058526992798,
                0.982741018
            ],
            [
                0.5807040333747864,
                0.984688443
            ],
            [
                0.586899995803833,
                0.986635869
            ],
            [
                0.592790961265564,
                0.988583295
            ],
            [
                0.5992965698242188,
                0.990530721
            ],
            [
                0.6073158383369446,
                0.992478147
            ],
            [
                0.6179296970367432,
                0.994425572
            ],
            [
                0.6316601037979126,
                0.996372998
            ],
            [
                0.6530047059059143,
                0.998320424
            ],
            [
                0.7702784538269043,
                0.999996849
            ]
        ]
    }
}
# --- END COLD_RANKING_CALIBRATION -------------------------------------

#: Cosine at or above which greedy leader clustering admits a node to an
#: existing cluster. A fallback stage's threshold, deliberately loose: whatever
#: reaches stage 4 has no structural signal at all, and a tight cut would
#: return one singleton per node — a map that says nothing, expensively.
EMBEDDING_CLUSTER_COSINE = 0.55

#: Distinct terms per cluster that reach the document-frequency index. Each is
#: one indexed seek; the top of a cluster's term-frequency list is where the
#: label lives anyway.
CTFIDF_TERM_BUDGET = 40

#: Terms in a c-TF-IDF label.
CTFIDF_LABEL_TERMS = 3

#: Distinct label terms the delivery gate scores. Capping the sum at three is
#: what makes :data:`LABEL_GATE_MIN_IC` mean anything: without it a label buys
#: its way past the floor by concatenating house vocabulary. Measured on the
#: field corpus, the four house words ``public``/``server``/``mmo``/``test``
#: sum to 4.191 across all four and to 3.619 across their best three — the cap
#: is the difference between a floor of 4.0 that admits them and one that does
#: not.
LABEL_GATE_TOP_TERMS = 3

#: Information content, in nats, a label's best :data:`LABEL_GATE_TOP_TERMS`
#: terms must carry for the cluster to be delivered at all.
#:
#: ``ic(t) = log((N + 1) / (1 + df(t)))`` against the same FTS
#: document-frequency index :meth:`RecallMapBuilder._ranked_terms` reads, so
#: "generic" means generic *to the corpus recall searches* and the rule
#: transfers to a corpus whose house vocabulary is different words. That is the
#: whole reason this is a statistic and not a word list: an enumerated
#: blacklist of the labels a field window happened to show would be a lookup
#: table wearing a threshold.
#:
#: Placed in a void rather than fitted to a boundary. Over every label the
#: field ever delivered (294 clusters, 21 distinct labels, ``N = 2573``) the
#: single-content-term population scored 0.57–2.72 and the multi-word one
#: 4.89–15.53; the interval ``(2.72, 4.89)`` is empty and 4.0 sits inside it.
#:
#: The floor encodes the term-count half of the predicate implicitly, which is
#: better than stating it twice: a *solo* term clears 4.0 only when it appears
#: in ``(N + 1) / e**4 - 1`` documents — 1.79 % of the corpus. A one-word label
#: is not banned by rule, it is admitted exactly when one word genuinely is
#: that rare.
LABEL_GATE_MIN_IC = 4.0

#: Tokens an enriched ask-hint may carry on the path and structural stages.
#:
#: Not a style rule — the whole reason label enrichment does not silently
#: retune :data:`CURTAIL_QUERY_OVERLAP`. ``_echoes`` needs ``ceil(n/2)`` shared
#: tokens for a phrasing of ``n``, and ``ceil(1/2) == ceil(2/2) == 1``: a hint
#: widened from one token to two asks a later query for exactly the evidence it
#: asked for before, while a three-token hint would ask for two. Replayed over
#: 2 524 (delivery, later-query) pairs in the field window, holding the hint at
#: two tokens is bit-identical to baseline (688 echo fires, 67 of 77 deliveries
#: judged followed) and letting it follow a three-token label destroys every
#: one of them.
#:
#: The cap governs *enrichment*; it never shortens a hint that was already
#: longer, because dropping a token breaks the same containment guarantee it
#: exists to protect. Stage 3 is exempt outright — an anchor's hint is a
#: question that already worked and is never rewritten.
ASK_HINT_MAX_TOKENS = 2

#: Name triples the filter journal carries. Four, because the counts beside
#: them are the honesty guarantee and the names are the convenience: a reader
#: who needs more than four examples of what was filtered is reading the
#: persisted payloads, not the live map.
FILTER_JOURNAL_NAMES = 4

#: Journal names are shortened harder than delivered labels, for the same
#: reason ``instructions_map`` shortens to 28: a journal name is a forensic
#: breadcrumb, not an invitation, and it competes for the same
#: :data:`MAX_RESPONSE_CHARS` as the medoid examples that are.
FILTER_JOURNAL_LABEL_CHARS = 24

#: Share of the current pool a cached structure must still cover to be reused.
#: Below it the cached labels describe a pool that has moved on, and stability
#: would be preserved at the cost of describing nothing.
CACHE_MIN_COVERAGE = 0.5

#: Distinct ``(scope, task)`` keys kept. One session touches a handful.
MAX_CACHE_ENTRIES = 32

#: Consecutive unconsumed deliveries under one key before the map collapses.
#: Three, because two is inside the noise of a single distracted session and
#: four spends a fourth full map to learn what the third already said. The
#: streak counts *offers* — deliveries that actually carried clusters — so the
#: collapsed markers a curtailed key keeps emitting can never push it up.
CURTAIL_STREAK = 3

#: Deliveries the curtail probe reads back — a *window*, not a history, and
#: the width of it is what sets the retry.
#:
#: A collapse is remembered only for as long as the offers that caused it stay
#: inside this window. The collapsed markers that follow them push them out,
#: and when the last of them goes the key offers full maps again: with
#: :data:`CURTAIL_STREAK` of 3, a key that is never consumed spends 3
#: deliveries of every 25 on a map and the other 22 on a ~60-character marker.
#: That 12% duty cycle is deliberate. A window of exactly the streak would
#: forget its own reason on the very next delivery and never collapse at all;
#: an unbounded one would make the first three unread maps a life sentence,
#: and a task's corpus — and the agent working it — both move.
CURTAIL_HISTORY_LIMIT = 24

#: Share of a cluster's label (or ask-hint) tokens a later query must carry for
#: that cluster to count as asked about. Half: a two-token label needs both, a
#: four-token one needs two, and a query that merely shares the corpus's house
#: vocabulary with a cluster does not clear it.
CURTAIL_QUERY_OVERLAP = 0.5

#: Valve for the decaying curtail retry, drain-valve spelling like the pool
#: gates. Off — unset, empty, ``0``, a typo — is the frozen cadence above,
#: byte for byte. On, a collapsed key retries after
#: ``min(2**(offers-1), CURTAIL_DECAY_SKIP_MAX)`` collapsed deliveries instead
#: of waiting for the window to expire its evidence. Registered in
#: ``~/p/ae/artifacts/injection-throttle/prereg-draft.md`` (П2), off a field
#: number: the quiet armored day on the alt store that first filled the maps
#: also pushed their curtailed share 0.024 → 0.562.
CURTAIL_DECAY_GATE_ENV = "LM_MAP_CURTAIL_DECAY"

#: Ceiling of the decay's skip run, from the registered rule
#: ``min(2**(N-1), 8)``: at the entry threshold of 3 the first run is 4, one
#: more unaccepted retry doubles it to 8, and there it stays. At the plateau a
#: key that is never followed spends 1 delivery in 9 on a map — the same
#: budget as the frozen 3-in-25 — but its first retry lands 4 deliveries
#: after the collapse instead of 22.
CURTAIL_DECAY_SKIP_MAX = 8

#: Hex characters of the pool digest a decay-armed marker persists (``pd``).
#: 64 bits: the comparison is one key's window against itself, and the marker
#: has ~640 of its 700 characters to spare.
CURTAIL_POOL_DIGEST_CHARS = 16

#: Context fields consulted by stage 1, in precedence order.
STRUCTURAL_FIELDS: tuple[str, ...] = (
    "procedure_id",
    "lesson_kind",
    "type",
    "topic",
    "task_pattern",
)

#: Context fields stage 2 reads paths out of.
_PATH_FIELDS: tuple[str, ...] = (
    "files",
    "file",
    "paths",
    "path",
    "files_seen",
    "files_owned",
    "files_read",
    "files_evidence",
)

#: Directory segments that name a *container*, not a subsystem. Stage 2 starts
#: reading after the last one it sees, which is what turns
#: ``/home/u/p/lm/src/living_memory/postsession/x.py`` into ``postsession``
#: rather than into ``u``.
_PATH_ROOT_SEGMENTS = frozenset(
    {"src", "lib", "app", "pkg", "internal", "cmd", "source", "sources", "tests", "test"}
)

STAGE_STRUCTURAL = "structural"
STAGE_PATH = "path"
STAGE_ANCHOR = "anchor"
STAGE_EMBEDDING = "embedding"
#: The appended cold-lane singletons. Not a cascade stage: cold clusters are
#: never produced by :meth:`RecallMapBuilder._cluster`, never cached, never
#: sorted into the warm order, and absent from :data:`_STAGE_ORDER` on
#: purpose — they exist only past the warm cap.
STAGE_COLD = "cold"

#: Precedence, and the secondary sort key for equal-sized clusters: a cluster
#: whose label is a key the corpus actually stores outranks one whose label was
#: derived from term statistics.
_STAGE_ORDER: dict[str, int] = {
    STAGE_STRUCTURAL: 0,
    STAGE_PATH: 1,
    STAGE_ANCHOR: 2,
    STAGE_EMBEDDING: 3,
}

#: Which filter took a cluster off the wire. One character each, because they
#: ride in the payload beside the names they qualify, and the distinction they
#: carry is the one a forensic reader actually needs: ``w`` says the label gate
#: refused to deliver the cluster at all, ``d`` says the response budget could
#: not afford it. "The filter is too harsh" and "the budget is tight" are
#: different findings and must not be read off one number.
FILTER_TAG_WITHHELD = "w"
FILTER_TAG_DROPPED = "d"

_WHITESPACE_RE = re.compile(r"\s+")
_TERM_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_HEX_DIGITS = frozenset("0123456789abcdef")
_VOWELS = frozenset("aeiouyаеёиоуыэюя")
_STRATEGY_STAGNATION_RE = re.compile(
    r"^Strategy stagnation detected on [^\r\n]{1,160}(?:\r?\n"
    r"(?:attempts|strategy|window|reason):[^\r\n]{1,160}){0,4}\s*$"
)
_FILE_CHUNK_INDEX_RE = re.compile(r"^[1-9][0-9]*/[1-9][0-9]*$")
_FILE_CHUNK_LINES_RE = re.compile(r"^[1-9][0-9]*-[1-9][0-9]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_JOURNAL_KINDS = frozenset(
    {
        "supervision_journal",
        "monitoring_journal",
        "execution_journal",
        "goal_tree_journal",
    }
)
_JOURNAL_METADATA_KEYS = ("kind", "type", "lesson_kind", "record_kind")


# ----------------------------------------------------------------------
# The map
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MapMedoid:
    """The member that speaks for a cluster, plus a short quote from it."""

    node_id: str
    example: str

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"node_id": self.node_id}
        if self.example:
            payload["example"] = self.example
        return payload


@dataclass(frozen=True, slots=True)
class MapCluster:
    """One row of the map: a label, how much of it there is, and how to ask."""

    label: str
    count: int
    medoid: MapMedoid
    #: How to phrase a recall for this cluster. At stage 3 this is a past
    #: grounded query verbatim — not a guess at good wording, but wording that
    #: already worked once.
    ask_hint: str
    #: Transferable into the agent's own todo plan, which on a linear
    #: ``--print`` pass is its only surface of self-observation.
    plan_item: str
    stage: str
    member_ids: tuple[str, ...]
    #: Set only on clusters the cold exploration lane appended. The additive
    #: wire marker is what lets after-window tooling split medoid delivery
    #: windows into warm and cold arms — ``recall_delivery_history`` rows carry
    #: no lane provenance and must not be extended to.
    cold: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "label": self.label,
            "count": self.count,
            "medoid": self.medoid.to_dict(),
            "ask_hint": self.ask_hint,
            "plan_item": self.plan_item,
        }
        if self.cold:
            payload["cold"] = 1
        return payload


@dataclass(frozen=True, slots=True)
class FilteredCluster:
    """One cluster the map did not deliver, reduced to what a reader can use.

    A name and a size, tagged with which filter took it. Deliberately *not* a
    :class:`MapCluster`: a filtered cluster has no medoid, no ask-hint and no
    plan item, because there is nothing here for the agent to act on — this is
    forensics, and it lives in its own payload key precisely so that no
    consumer can mistake it for something on offer.
    """

    #: :data:`FILTER_TAG_WITHHELD` or :data:`FILTER_TAG_DROPPED`.
    tag: str
    label: str
    count: int

    def to_list(self) -> list[Any]:
        """The wire form: a triple, not an object, to save the JSON keys."""

        return [self.tag, self.label, self.count]


@dataclass(frozen=True, slots=True)
class SelectionSample:
    """One identity-free example of a member-level exclusion.

    ``ordinal`` is retained only while fitting the additive selection block;
    it is never serialized.  It makes the choice independently auditable
    without turning a node id into policy input or wire content.
    """

    reason: str
    gist: str
    ordinal: int

    def to_list(self) -> list[str]:
        return [self.reason, self.gist]


@dataclass(frozen=True, slots=True)
class SelectionAccounting:
    """Complete frozen-policy accounting for one inspected residual.

    The response fitter added by the delivery integration can vary only how
    many of ``samples`` it buys.  ``inspected``, ``admitted`` and ``excluded``
    are unconditional, and ``to_dict`` recomputes ``o`` from the chosen sample
    count so dropping examples can never falsify the accounting equation.

    ``excluded`` is the frozen :data:`SELECTION_REASON_CODES` vector,
    optionally extended by the :data:`POOL_GATE_REASON_CODES` an env valve
    turned on.  The extension is *canonically trimmed*: a trailing zero past
    the frozen width is rejected rather than accepted-and-ignored, so there is
    exactly one encoding of "no gate fired" and it is byte-identical to what a
    build with no gates at all produced.  That is the invariant, not a
    convention — an accounting that could spell the same fact two ways would
    make the byte-identity claim unfalsifiable.
    """

    inspected: int
    admitted: int
    excluded: tuple[int, ...]
    samples: tuple[SelectionSample, ...] = ()
    sampleable: int = 0

    def __post_init__(self) -> None:
        frozen_width = len(SELECTION_REASON_CODES)
        width = len(self.excluded)
        if not frozen_width <= width <= len(SELECTION_LEDGER_REASON_CODES):
            raise ValueError("selection exclusion vector has the wrong width")
        if width > frozen_width and not self.excluded[-1]:
            raise ValueError("selection exclusion vector is not canonically trimmed")
        if self.inspected < 0 or self.admitted < 0 or any(
            count < 0 for count in self.excluded
        ):
            raise ValueError("selection counts must be non-negative")
        if self.inspected != self.admitted + sum(self.excluded):
            raise ValueError("selection accounting does not cover the residual")
        if self.sampleable < len(self.samples):
            raise ValueError("selection samples exceed the sampleable population")
        sampleable_cap = sum(
            count
            for count, reason in zip(
                self.excluded, SELECTION_LEDGER_REASON_CODES, strict=False
            )
            if reason in _SAMPLEABLE_REASONS
        )
        if self.sampleable > sampleable_cap:
            raise ValueError("selection sampleable count exceeds exclusions")

        reason_order = {
            reason: index
            for index, reason in enumerate(SELECTION_LEDGER_REASON_CODES)
        }
        prior = -1
        seen: set[str] = set()
        for sample in self.samples:
            if sample.reason not in _SAMPLEABLE_REASONS:
                raise ValueError("selection sample has a non-sampleable reason")
            current = reason_order[sample.reason]
            if current >= width or self.excluded[current] <= 0:
                raise ValueError("selection sample has no matching exclusion")
            if current <= prior or sample.reason in seen:
                raise ValueError("selection samples are not in fixed reason order")
            if not sample.gist or len(sample.gist) > SELECTION_SAMPLE_GIST_CHARS:
                raise ValueError("selection sample gist is empty or over budget")
            if sample.ordinal < 0:
                raise ValueError("selection sample ordinal must be non-negative")
            prior = current
            seen.add(sample.reason)

    def to_dict(self, *, sample_limit: int = SELECTION_SAMPLE_LIMIT) -> dict[str, Any]:
        """Return the additive ``sel`` shape with a budgeted ``q``."""

        limit = min(SELECTION_SAMPLE_LIMIT, max(0, int(sample_limit)))
        chosen = self.samples[:limit]
        payload: dict[str, Any] = {
            "v": "r1",
            "n": self.inspected,
            "e": self.admitted,
            "x": list(self.excluded),
            "o": self.sampleable - len(chosen),
        }
        if chosen:
            payload["q"] = [sample.to_list() for sample in chosen]
        return payload


@dataclass(frozen=True, slots=True)
class RecallMap:
    """A capped, ordered set of clusters over one recall's residual pool."""

    scope: str
    #: The cache key this map was built or served under.
    key: str
    clusters: tuple[MapCluster, ...]
    #: Members of the residual pool the map considered.
    pool_size: int
    #: Members the clusters actually account for.
    covered: int
    #: Clusters the caps dropped. Reported, never silent.
    dropped: int = 0
    #: Clusters the label gate refused (:meth:`RecallMapBuilder._deliverable`).
    #: Deliberately *not* folded into ``dropped``: a harsh filter and a tight
    #: budget are different findings, and the field cannot tell them apart
    #: after the fact if the map reports one number for both.
    withheld: int = 0
    #: Names of what the filter removed, most reportable first, already cut to
    #: what :data:`MAX_RESPONSE_CHARS` afforded.
    filtered: tuple[FilteredCluster, ...] = ()
    #: Names the budget did not afford. The anti-silence field: the counts
    #: above are unconditional, so the map never drops a cluster without a
    #: number attached, and this says how many of them went unnamed.
    filtered_omitted: int = 0
    #: Set when this key's last :data:`CURTAIL_STREAK` maps went unconsumed.
    #: A curtailed map carries no clusters: it has stopped describing the pool
    #: and started reporting that describing it was not worth the channel.
    curtailed: bool = False
    #: Consecutive unconsumed deliveries behind the collapse. *The* signal —
    #: the number is what tells an operator (and the effect gate) that the map
    #: went quiet and how long it has been quiet for.
    streak: int = 0
    #: Member eligibility and relevance selection, computed before clustering.
    #: It travels with every inspected residual so no later layer has to
    #: reconstruct unconditional counts from the capped pool.
    selection: SelectionAccounting | None = None
    #: Optional ``q`` examples that survived the response budget.  The core is
    #: never conditional; only these identity-free examples may be removed.
    selection_sample_limit: int = SELECTION_SAMPLE_LIMIT
    #: ``(g, k)`` of the cold exploration lane — cold-eligible candidates
    #: examined and cold clusters delivered — present exactly when the cold
    #: valve pair is armed, even at ``(0, 0)``: an armed-and-idle lane and an
    #: unarmed lane are different facts and the wire must not spell them the
    #: same way. ``None`` (the valves-off value) emits nothing, so absence of
    #: the lane stays encoded exactly one way.
    cold: tuple[int, int] | None = None
    #: Digest of the pool this delivery was decided over (``pd``), present on
    #: exactly the deliveries a decay-armed key uses as a baseline: the first
    #: marker after an offer, and a served skip run that rebuilt the pool and
    #: found nothing new in it. ``None`` — the valve-unset value — emits
    #: nothing, so the frozen marker stays byte-identical.
    pool_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """The response-shaped form, budgeted by :data:`MAX_RESPONSE_CHARS`."""

        return _payload(
            self.clusters,
            self.pool_size,
            self.dropped,
            curtailed=self.curtailed,
            streak=self.streak,
            filtered=_filter_block(
                self.withheld, self.dropped, self.filtered, self.filtered_omitted
            ),
            selection=self.selection,
            selection_sample_limit=self.selection_sample_limit,
            cold=self.cold,
            pool_digest=self.pool_digest,
        )

    def plan_items(self) -> list[str]:
        return [cluster.plan_item for cluster in self.clusters]

    def render_compact(self) -> str:
        """One line, at most :data:`MAX_INSTRUCTIONS_CHARS` chars.

        Labels and counts only: the instructions channel has room for the
        *shape* of what memory holds, and the phrasings live one recall away.
        Clusters are dropped from the tail until the line fits, so the first —
        largest — cluster survives any budget.

        A curtailed map carries no clusters and so renders as the empty string,
        which is the entire point of the collapse: the channel it was spending
        is handed back rather than spent on a line nobody read.
        """

        prefix = "memory also holds: "
        parts: list[str] = []
        for cluster in self.clusters:
            candidate = [*parts, f"{cluster.label}({cluster.count})"]
            if len(prefix + " · ".join(candidate)) > MAX_INSTRUCTIONS_CHARS:
                break
            parts = candidate
        if not parts:
            return ""
        return prefix + " · ".join(parts)


# ----------------------------------------------------------------------
# Internal working shapes
# ----------------------------------------------------------------------


@dataclass(slots=True)
class _Member:
    """One admitted entry with its frozen score and original residual ordinal."""

    rank: int
    node: Node
    relevance_score: float = 0.0


@dataclass(slots=True)
class _SelectionLedger:
    """Mutable accounting while the complete residual is classified."""

    inspected: int
    excluded: Counter[str] = field(default_factory=Counter)
    first_samples: dict[str, SelectionSample] = field(default_factory=dict)
    sampleable: int = 0

    def exclude(self, reason: str, *, ordinal: int, node: Any = None) -> None:
        self.excluded[reason] += 1
        if reason not in _SAMPLEABLE_REASONS or node is None:
            return
        gist = _selection_gist(node)
        if not gist:
            return
        self.sampleable += 1
        self.first_samples.setdefault(
            reason, SelectionSample(reason=reason, gist=gist, ordinal=ordinal)
        )

    def freeze(self, admitted: int) -> SelectionAccounting:
        """Close the ledger into the canonical, trimmed accounting.

        The exclusion vector is built over every code this module knows and
        then cut back to :data:`SELECTION_REASON_CODES` while its tail is
        zero. With both pool valves unset that cut always reaches the frozen
        width, so the payload is the one this module emitted before the gate
        codes existed — down to the length of ``x``.
        """

        samples = tuple(
            self.first_samples[reason]
            for reason in SELECTION_LEDGER_REASON_CODES
            if reason in self.first_samples
        )
        counts = [self.excluded[reason] for reason in SELECTION_LEDGER_REASON_CODES]
        while len(counts) > len(SELECTION_REASON_CODES) and not counts[-1]:
            counts.pop()
        return SelectionAccounting(
            inspected=self.inspected,
            admitted=admitted,
            excluded=tuple(counts),
            samples=samples,
            sampleable=self.sampleable,
        )


@dataclass(slots=True)
class _Group:
    """A cluster under construction, before labels are trimmed and capped."""

    stage: str
    #: Membership rule, and the cache's identity for this cluster. See
    #: :meth:`RecallMapBuilder._recount`.
    signature: tuple[Any, ...]
    label: str
    ask_hint: str
    members: list[_Member]
    #: Set only by stage 4, which is the only stage holding vectors.
    vectors: dict[str, list[float]] | None = None


@dataclass(slots=True)
class _FilterJournal:
    """What the filter removed on one build, before the budget names it.

    Two populations, kept apart all the way to the wire. They answer different
    questions — ``withheld`` says the gate found nothing worth saying,
    ``dropped`` says the channel had no room for something that was — and the
    no-silent-caps rule is only satisfied if a field reader can tell which.

    Order is the report order: the gate's refusals first, largest first, then
    the budget's drops newest-first. Both halves therefore lead with the
    cluster whose removal cost the reader most, which is what survives when
    :data:`FILTER_JOURNAL_NAMES` takes the tail.
    """

    withheld: list[FilteredCluster] = field(default_factory=list)
    dropped: list[FilteredCluster] = field(default_factory=list)
    #: Names are forensic convenience, not the honesty guarantee.  ``_fit``
    #: lowers this only after breadth reaches its floor and before gist does.
    name_limit: int = FILTER_JOURNAL_NAMES

    def entries(self) -> list[FilteredCluster]:
        return [*self.withheld, *self.dropped]

    def names(self) -> tuple[FilteredCluster, ...]:
        return tuple(self.entries()[: self.name_limit])

    def omitted(self) -> int:
        return max(0, len(self.entries()) - len(self.names()))

    def drop_name(self) -> bool:
        """Give one currently visible name to the gist budget."""

        visible = len(self.names())
        if visible <= 0:
            return False
        self.name_limit = visible - 1
        return True

    def block(self) -> dict[str, Any] | None:
        return _filter_block(
            len(self.withheld), len(self.dropped), self.names(), self.omitted()
        )


@dataclass(frozen=True, slots=True)
class _ClusterTemplate:
    """A cached cluster: its identity and its labels, without its counts."""

    stage: str
    signature: tuple[Any, ...]
    label: str
    ask_hint: str
    member_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class _CachedStructure:
    """One cache entry: templates plus the corpus revision that validates them."""

    revision: tuple[Any, ...]
    #: Empty only for a freshly selected empty pool.  This is a revision-bound
    #: negative structure result, not a wildcard: a later non-empty pool must
    #: rebuild rather than treating it as a reusable structure.
    templates: tuple[_ClusterTemplate, ...]


@dataclass(frozen=True, slots=True)
class _DeliveredItem:
    """One cluster of an already-delivered map, reduced to what proves use.

    Read back out of ``recall_events.recall_map``, so the fields are whatever
    :meth:`MapCluster.to_dict` wrote and nothing else: the two phrasings a
    later query can echo, and the node id a later delivery can reach.
    """

    medoid_id: str
    label: frozenset[str]
    ask_hint: frozenset[str]


@dataclass(frozen=True, slots=True)
class _Curtailment:
    """How long this key's map has been delivered without being followed."""

    #: Consecutive most-recent deliveries with nothing consumed, collapsed
    #: markers included. Reported to the agent and to the effect gate.
    streak: int
    #: Of those, the ones that actually offered clusters. Only an offer can go
    #: unaccepted, so only an offer may push the map towards silence.
    offers: int
    #: The newest deliveries of the streak that offered nothing — how far the
    #: key is into its current run of markers, zeroed the moment a map goes
    #: out. Only :attr:`retry_due` reads it; a verdict served from the memo
    #: carries the default, which is safe because the memo never serves a
    #: collapse and only a collapse can be retried.
    lead: int = 0
    #: The pool digest the newest lead row carries, when one does: the
    #: baseline a decay-armed retry compares the rebuilt pool against. Written
    #: by the first marker after an offer and by every retry that found
    #: nothing new (:meth:`RecallMapBuilder._reoffer`), read back off the same
    #: window the streak is, so it survives a restart and is shared by every
    #: process serving the key. ``None`` on a window written with the valve
    #: unset — which is every window until the valve is armed — and on one
    #: whose digest rows have all slid out.
    pool_digest: str | None = None
    #: Lead rows newer than the one carrying :attr:`pool_digest`. Counting the
    #: digest row itself, this is how much of the skip run the key has served
    #: since it last looked at its pool.
    since_digest: int = 0

    @property
    def collapse(self) -> bool:
        return self.offers >= CURTAIL_STREAK

    @property
    def retry_due(self) -> bool:
        """Whether a decay-armed key has served its skip run.

        Pure arithmetic on the verdict: the valve itself is read at the one
        collapse decision (:meth:`RecallMapBuilder._reoffer`), never here or
        below, so the probe path stays env-free and the unarmed decision is
        bit-identical to the frozen rule. The run is the registered
        ``min(2**(N-1), 8)`` with ``N`` the unaccepted offers — 4 at the
        entry threshold, 8 from the fourth on — and it is measured against a
        real read, because every collapse verdict comes off one.

        The run is counted from the last time the key looked at its pool: the
        digest row when the window carries one (the first marker after an
        offer, or a retry that found nothing new), the head of the lead when
        it carries none, which is every window the frozen code wrote and so
        the frozen schedule exactly.
        """

        if not self.collapse:
            return False
        served = self.lead if self.pool_digest is None else self.since_digest + 1
        return served >= min(2 ** (self.offers - 1), CURTAIL_DECAY_SKIP_MAX)


@dataclass(slots=True)
class _CurtailMemo:
    """A read verdict, plus what this builder has delivered since reading it.

    The history read is the one genuinely expensive thing on this path and it
    is expensive for a structural reason that no caller can fix: no index
    covers ``(scope, task, recall_map IS NOT NULL)`` in time order, so SQLite
    walks the whole scope partition and sorts — measured at 8.5 ms on a
    21k-event scope, on a recall whose median is 100 ms, *and rising with the
    partition*.

    So the read is asked only when its answer could matter. A collapse needs
    :data:`CURTAIL_STREAK` unaccepted offers; this builder knows how many
    offers it has made since it last read, and while the last read plus that
    count is still under the threshold, no history the read could return would
    collapse anything. Counting only forward, never back, is what makes the
    shortcut safe in the one direction that matters: another process's
    deliveries are invisible to it, so it can *under*-count and skip a
    collapse, and consumption is invisible to it, so it can *over*-count and
    ask for a read that turns out unnecessary. It can never collapse a map on
    its own word — every collapse in this module is decided by a read.

    A *confirmed* collapse is deliberately not held. The estimate stays at the
    threshold, so every collapsed delivery reads again — which is exactly what
    lets consumption reopen the channel on the very next recall instead of
    some deliveries later. It is affordable because a collapsed build skips
    the whole label cascade it replaces: 8.5 ms of read against the ~12 ms of
    clustering it no longer does. Measured over a replayed server loop on that
    21k-event scope: a key that is being followed spends 3.2 ms per delivery
    on this probe and answers two builds in three from memory, and a key that
    is not gets *faster* than it was before the rule existed.
    """

    verdict: _Curtailment
    #: Deliveries this builder has made under the key since the read, that
    #: carried clusters — the ones capable of going unaccepted.
    offers: int = 0
    #: All of them, collapsed markers included.
    total: int = 0
    #: Medoids of this key's window whose label or ask-hint a later query
    #: echoed — the ask-echo probe, harvested from the read that was happening
    #: anyway so the pool gate costs no second walk of the same rows.
    ask_followed: frozenset[str] = frozenset()
    #: Whether the read that seeded this memo was asked to harvest them. A
    #: memo seeded without them cannot serve the gate: an empty set would read
    #: as "nobody asked", which is the one direction that costs a row its
    #: place in the pool.
    ask_collected: bool = False

    def estimate(self) -> _Curtailment:
        return _Curtailment(
            streak=self.verdict.streak + self.total,
            offers=self.verdict.offers + self.offers,
        )


# ----------------------------------------------------------------------
# Normalization, mirrored deliberately
# ----------------------------------------------------------------------


def normalize_key(value: str) -> str:
    """Fold a slug-shaped identifier into readable words.

    Mirrors ``consolidation._normalize_trigger`` (consolidation.py:820) rather
    than importing it: this module's whole point is to be cheap and free of the
    consolidation stack, and the transform is four characters of policy. The
    mirror is pinned against the original in ``tests/test_recall_map.py`` so it
    cannot drift silently.
    """

    cleaned = str(value).replace("_", " ").replace("-", " ").replace("/", " ")
    return _WHITESPACE_RE.sub(" ", cleaned).strip().lower()


def _looks_unreadable(value: str) -> bool:
    """Whether a raw structural key is a hash rather than a name.

    ``task_pattern`` is routinely a content hash, and a hash makes a fine
    grouping key and a useless label. Three signals, all conservative: a long
    pure-hex run, a string that is half digits, and a long unpronounceable run
    with no vowel in it. Anything containing a space has already survived
    normalization as words and is left alone.
    """

    if not value:
        return True
    if " " in value:
        return False
    lowered = value.lower()
    if len(lowered) < 8:
        return False
    if len(lowered) >= 12 and all(char in _HEX_DIGITS for char in lowered):
        return True
    digits = sum(1 for char in lowered if char.isdigit())
    if digits * 2 >= len(lowered):
        return True
    return len(lowered) >= 16 and not any(char in _VOWELS for char in lowered)


def _collapse(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(text)).strip()


def _shorten(text: str, limit: int) -> str:
    collapsed = _collapse(text)
    if limit <= 0:
        return ""
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _utc_now() -> str:
    """One canonical map-build instant, captured before any history read."""

    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _is_file_chunk_envelope(envelope: Any) -> bool:
    """The evaluator's strict file-chunk schema, with no provenance lookup."""

    if not isinstance(envelope, dict):
        return False
    chunk = envelope.get("chunk")
    lines = envelope.get("lines")
    if not isinstance(chunk, str) or _FILE_CHUNK_INDEX_RE.fullmatch(chunk) is None:
        return False
    if not isinstance(lines, str) or _FILE_CHUNK_LINES_RE.fullmatch(lines) is None:
        return False
    part, total = (int(value) for value in chunk.split("/"))
    first_line, last_line = (int(value) for value in lines.split("-"))
    return (
        part <= total
        and first_line <= last_line
        and isinstance(envelope.get("path"), str)
        and bool(envelope["path"].strip())
        and isinstance(envelope.get("kind"), str)
        and bool(envelope["kind"].strip())
        and isinstance(envelope.get("language"), str)
        and isinstance(envelope.get("sha256"), str)
        and _SHA256_RE.fullmatch(envelope["sha256"]) is not None
    )


def _ballast_reason(
    content: Any,
    context: Any = None,
    provenance: Any = None,
) -> str | None:
    """Return ``fc``/``ss``/``sj`` for an exact machine envelope.

    Every parsing and shape failure is eligible.  In particular, marker words
    embedded in prose and journal names under ``topic`` are not classifiers.
    This function reads only the candidate's own form/schema/provenance; it has
    no emitter, host, project, task, label or identity surface.
    """

    if not isinstance(content, str):
        return None

    if content.startswith("[file-chunk]"):
        suffix = content[len("[file-chunk]") :]
        separator = suffix[:1]
        header = ""
        if separator in {" ", "\t"}:
            lines = suffix.lstrip(" \t").splitlines()
            header = lines[0].strip() if lines else ""
        try:
            envelope = json.loads(header)
        except (TypeError, ValueError):
            envelope = None
        if _is_file_chunk_envelope(envelope):
            return "fc"

    if _STRATEGY_STAGNATION_RE.fullmatch(content):
        return "ss"

    kind_values: set[str] = set()
    for mapping in (context, provenance):
        if not isinstance(mapping, Mapping):
            continue
        for key in _JOURNAL_METADATA_KEYS:
            try:
                value = mapping.get(key)
            except (AttributeError, TypeError, ValueError):
                continue
            kind_values.add(str(value or "").strip().lower().replace("-", "_"))
    first_line = content.splitlines()[:1]
    marker = (first_line[0] if first_line else "").strip().lower().replace("-", "_")
    if kind_values & _JOURNAL_KINDS or marker in {
        "[supervision_journal]",
        "[monitoring_journal]",
    }:
        return "sj"
    return None


def _history_features(history: Any) -> tuple[float, float, float, float]:
    """Frozen evaluator features, or train means when history is unavailable."""

    unavailable = (
        RELEVANCE_FEATURE_MEANS[1],
        RELEVANCE_FEATURE_MEANS[2],
        RELEVANCE_FEATURE_MEANS[3],
        RELEVANCE_FEATURE_MEANS[4],
    )
    if history is None or getattr(history, "available", False) is not True:
        return unavailable

    values = (
        getattr(history, "matured", None),
        getattr(history, "consumed", None),
        getattr(history, "trailing_nonconsumed", None),
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        return unavailable
    matured, consumed, streak = values
    if (
        matured < 0
        or consumed < 0
        or consumed > matured
        or streak < 0
        or streak > matured - consumed
    ):
        return unavailable

    # ``round(..., 8)`` is part of the unchanged evaluator's feature
    # arithmetic.  Keeping it here is necessary to reproduce threshold ties.
    return (
        round(log1p(matured), 8),
        round(log1p(matured - consumed), 8),
        round(log1p(streak), 8),
        round(consumed / matured, 8) if matured else 0.0,
    )


#: The candidate-time evidence a residual entry carries about *this* query, in
#: the order ``retrieval.RecallResult`` declares it (see
#: ``src/living_memory/retrieval.py``): the composite score first, then the four
#: components behind it. A node with no delivery history at all still owns every
#: one of these numbers for the query that just ranked it — which is exactly why
#: they are the material a cold-start feature family can be built from, and why
#: they are worth carrying to the scorer before any such family exists.
_CANDIDATE_SCORE_FIELDS: tuple[str, ...] = (
    "score",
    "bm25_score",
    "vector_score",
    "graph_score",
    "trigger_score",
)


def _candidate_scores(result: Any) -> tuple[float, ...] | None:
    """This candidate's own query-time scores, or ``None`` if unreadable.

    Fail-closed, and the closure is registered rather than invented:
    ``artifacts/recall-map/relevance/coldstart-prereg.json``'s
    ``missing_value_rule.cause_a_live_unreadable`` rules that a row whose own
    candidate-time evidence cannot be read as finite floats has, by definition,
    no candidate-time evidence to be admitted on. Such a row is inadmissible
    under the frozen ``lr`` code — the reason vector gains nothing, loses
    nothing and is not reordered for it.

    Absent and malformed deliberately collapse to the same answer: a caller
    that passes no result and a caller that passes a shape this cannot read are
    stating the same fact, so the absent case needs no sentinel of its own.

    Nothing reads this yet. :data:`RELEVANCE_POLICY_ID` spends five features
    and none of them is here, so today this reads evidence the scorer declines.
    Spending it is a policy revision, and that revision moves
    :data:`RELEVANCE_POLICY_DIGEST` with it.
    """

    values: list[float] = []
    for name in _CANDIDATE_SCORE_FIELDS:
        try:
            value = getattr(result, name, None)
        except (AttributeError, TypeError, ValueError):
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if not isfinite(number):
            return None
        values.append(number)
    return tuple(values)


def _relevance_features(
    node: Any, history: Any, result: Any = None
) -> tuple[float, ...]:
    """The frozen five, plus the candidate evidence no term spends yet.

    ``result`` is this candidate's ``RecallResult`` for this query, threaded
    here so the vector can one day be built from what a cold node actually
    owns. Under ``directional-zsum-r1`` it contributes nothing: the width stays
    five and every element is what it was, whether ``result`` is present,
    absent or malformed.
    """

    level = getattr(node, "level", None)
    level_schema = (
        float(level == "schema")
        if level in {"trace", "concept", "schema"}
        else RELEVANCE_FEATURE_MEANS[0]
    )
    return (level_schema, *_history_features(history))


def _directional_zsum(features: Sequence[float]) -> float:
    """Exact equal-weight directional-zsum-r1 arithmetic."""

    if len(features) != len(RELEVANCE_FEATURE_MEANS):
        raise ValueError("relevance feature vector has the wrong width")
    return sum(
        (float(value) - mean) / scale
        for value, mean, scale in zip(
            features,
            RELEVANCE_FEATURE_MEANS,
            RELEVANCE_FEATURE_SCALES,
            strict=True,
        )
    )


def relevance_score(node: Any, history: Any, result: Any = None) -> float:
    """Score one node from only its level and strictly matured history.

    ``result`` — the residual entry this node arrived in, holding the
    candidate-time scores :func:`_candidate_scores` reads — is accepted and
    deliberately unread. Carrying it is what lets a cold-start revision change
    the *scorer* without touching the call site that feeds it; making it count
    is that revision, and it moves :data:`RELEVANCE_POLICY_ID` and
    :data:`RELEVANCE_POLICY_DIGEST` together. Until then
    ``relevance_score(node, history)`` and
    ``relevance_score(node, history, result)`` are the same float for every
    input, bit for bit.
    """

    return _directional_zsum(_relevance_features(node, history, result))


# ----------------------------------------------------------------------
# Rank calibration: reading a published table, and looking a value up in it
# ----------------------------------------------------------------------
#
# ``directional-zsum-r1`` calibrates nothing.  A cold-start revision cannot
# keep that up: its arms are a history z-sum and a family of retrieval scores,
# quantities with no common unit, and summing them raw would let whichever one
# happens to be numerically larger decide every row.
# ``artifacts/recall-map/relevance/coldstart-prereg-v2.json`` registers the
# answer in advance, at ``scorer_shape.calibration``: each arm is put on a rank
# scale by a monotone step function fitted on ``coldstart_train`` alone,
# published in the policy, and evaluated by *lookup* everywhere else.  "The
# published step function IS the definition" — fit, eval, holdout and the live
# scorer must all read the same table, or a threshold tie stops reproducing.
#
# This is the evaluator for that table, and it is landed empty-handed on
# purpose.  It owns the form — how a table is read, and how a value is looked
# up in one — and none of the content: every float it can return was handed to
# it.  :data:`RELEVANCE_POLICY_CALIBRATION` is empty, so nothing calls it and
# no score moves.  What that buys is that the revision commit has one job,
# spending a table, instead of two.


@dataclass(frozen=True)
class _CalibrationTable:
    """One published monotone step function, validated once when read.

    ``values`` and ``us`` are the *merged* breakpoints — atoms and knots
    together, sorted by value, strictly increasing in value and non-decreasing
    in ``u``.  Merging them is what makes the composite monotone: an atom held
    apart as a pure equality override may sit anywhere inside the bracket its
    neighbouring knots interpolate across, and a probe a hair below such an
    atom would then score *above* it.  Merged, the atom is a breakpoint the
    interpolation bends through, so the function is monotone everywhere and
    the atom is still recovered bit-for-bit by equality.

    ``atoms`` and ``knots`` are kept as published so a caller can tell the two
    apart afterwards — they are read back by the resolution rule below, and
    they are what a test can hold the evaluator to.
    """

    quantity: str
    values: tuple[float, ...]
    us: tuple[float, ...]
    atoms: tuple[tuple[float, float], ...]
    knots: tuple[tuple[float, float], ...]


def _calibration_number(value: Any, what: str) -> float:
    """One published float, or a refusal naming what was wrong with it."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} is not a real number: {value!r}")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{what} is not finite: {value!r}")
    return number


def _calibration_pairs(raw: Any, what: str) -> tuple[tuple[float, float], ...]:
    """A published ``(value, u)`` list, refusing anything that is not one."""

    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"{what} is not a sequence of value/u pairs")
    pairs: list[tuple[float, float]] = []
    for index, entry in enumerate(raw):
        if (
            isinstance(entry, (str, bytes))
            or not isinstance(entry, Sequence)
            or len(entry) != 2
        ):
            raise ValueError(f"{what}[{index}] is not a value/u pair")
        pairs.append(
            (
                _calibration_number(entry[0], f"{what}[{index}] value"),
                _calibration_number(entry[1], f"{what}[{index}] u"),
            )
        )
    for index in range(1, len(pairs)):
        if pairs[index][0] <= pairs[index - 1][0]:
            raise ValueError(
                f"{what} are not sorted by strictly increasing value "
                f"at index {index}"
            )
    return tuple(pairs)


def _calibration_table(calibration: Any, quantity: str) -> _CalibrationTable:
    """The published table for ``quantity``, or a refusal to guess one.

    Fail-closed in the one way that matters here: there is no neutral answer to
    fall back on.  ``rank_uniform_centered`` is centred at the fitting median,
    so its neutral value is ``0.0`` — a number that means "this row sits where
    half the population sits".  Returning that for a table this could not read
    would not be a degraded answer, it would be a *fabricated measurement*,
    and the arm it feeds would go on being summed as if it had been calibrated.
    So every unreadable shape raises, and a caller that cannot produce a table
    has to say so upstream where the row can be dropped honestly.

    What is refused, and why each one is not recoverable:

    * no calibration block at all, or none for this quantity — the policy never
      fitted this arm, so there is nothing to look anything up in;
    * fewer than two knots — a one-knot grid interpolates nothing and clamps
      every input to a single constant, which is the neutral-value failure
      wearing a table's clothes;
    * a value or a ``u`` that is not a finite real — ``nan`` compares false
      against every bound, so it would slip through the ordering checks and
      then poison the sum;
    * knots or atoms not strictly increasing in value — the lookup is a binary
      search and an unsorted array silently answers the wrong bracket;
    * an atom sitting on a knot's value — two published answers for one input;
    * a merged pair that steps *down* in ``u`` — the function is not monotone,
      and monotonicity is the whole reason a rank scale preserves the arm's
      ordering;
    * more breakpoints than the registered form allows;
    * two consecutive knots further apart in ``u`` than
      :data:`RELEVANCE_CALIBRATION_MAX_INTERPOLATION_ERROR` with no atom
      between them — the grid is too coarse to honour the bound it is
      published under.

    That last check exempts knot pairs straddling an atom, and the exemption is
    structural rather than a concession: an atom is published exactly because
    its value carries at least 0.5% of the fitting rows, so the ``u`` it jumps
    across is at least ``0.005`` — wider than ``2**-9`` by construction.  No
    grid can resolve that jump, which is why the registered form carries the
    value exactly instead of trying to.
    """

    if not isinstance(calibration, Mapping) or not calibration:
        raise ValueError("the policy publishes no calibration tables")
    raw = calibration.get(quantity)
    if raw is None:
        raise ValueError(f"the policy publishes no calibration table for {quantity!r}")
    if not isinstance(raw, Mapping):
        raise ValueError(f"the calibration table for {quantity!r} is not a mapping")

    knots = _calibration_pairs(raw.get("knots"), f"{quantity} knots")
    atoms = _calibration_pairs(raw.get("atoms", ()), f"{quantity} atoms")
    if len(knots) < 2:
        raise ValueError(
            f"the calibration table for {quantity!r} publishes fewer than two knots"
        )

    knot_values = {value for value, _u in knots}
    shared = sorted(value for value, _u in atoms if value in knot_values)
    if shared:
        raise ValueError(
            f"the calibration table for {quantity!r} publishes both an atom and "
            f"a knot at {shared[0]!r}"
        )

    breakpoints = sorted(atoms + knots)
    if len(breakpoints) > RELEVANCE_CALIBRATION_MAX_BREAKPOINTS:
        raise ValueError(
            f"the calibration table for {quantity!r} spends {len(breakpoints)} "
            f"breakpoints, over the registered "
            f"{RELEVANCE_CALIBRATION_MAX_BREAKPOINTS}"
        )
    for index in range(1, len(breakpoints)):
        (low_value, low_u), (high_value, high_u) = (
            breakpoints[index - 1],
            breakpoints[index],
        )
        if high_u < low_u:
            raise ValueError(
                f"the calibration table for {quantity!r} is not monotone: u steps "
                f"down from {low_u!r} at {low_value!r} to {high_u!r} at "
                f"{high_value!r}"
            )
        if not isfinite(high_value - low_value):
            raise ValueError(
                f"the calibration table for {quantity!r} spans a value gap that "
                f"is not representable, at {low_value!r}"
            )

    atom_values = tuple(value for value, _u in atoms)
    for index in range(1, len(knots)):
        (low_value, low_u), (high_value, high_u) = knots[index - 1], knots[index]
        if any(low_value < atom < high_value for atom in atom_values):
            continue
        if high_u - low_u > RELEVANCE_CALIBRATION_MAX_INTERPOLATION_ERROR:
            raise ValueError(
                f"the calibration table for {quantity!r} steps u by "
                f"{high_u - low_u!r} between the knots at {low_value!r} and "
                f"{high_value!r}, over the registered "
                f"{RELEVANCE_CALIBRATION_MAX_INTERPOLATION_ERROR}"
            )

    return _CalibrationTable(
        quantity=quantity,
        values=tuple(value for value, _u in breakpoints),
        us=tuple(u for _value, u in breakpoints),
        atoms=atoms,
        knots=knots,
    )


def _calibrate(table: _CalibrationTable, value: Any) -> float:
    """Look ``value`` up in a published table and return its ``u``.

    Three rules, in the order the registered form states them.  A value the
    table publishes is **matched by equality first**, before any arithmetic
    happens, and its published ``u`` is returned unchanged — that is what makes
    a tie with the fitting sample recover the midrank of its block bit-for-bit
    instead of an interpolation that lands an ulp away, and the tie-block
    identity the feasibility argument rests on needs the exact one.  A value
    between two breakpoints is linearly interpolated.  A value outside the
    outermost breakpoints is clamped to the outermost published ``u``: the
    fitting sample never went there, and the honest reading of "further out
    than anything I was fitted on" is the extreme rank, not an extrapolation
    off the end of the evidence.

    The interpolation is clamped into its own bracket, and that clamp is not
    idle.  On a table whose values span wide enough,
    ``(probe - low) / (high - low)`` rounds to exactly ``1.0`` for a probe
    strictly *below* the segment's top, and ``low_u + (high_u - low_u)`` need
    not round back to ``high_u`` — so unclamped, a segment can end above the
    breakpoint the next one starts at, and a table the reader certified as
    monotone evaluates non-monotonically for purely floating-point reasons.
    The reader constrains a span to be representable, not to be narrow, so
    that table is one it accepts.  Everything else here is monotone by
    construction: a correctly rounded subtraction, a division by a positive
    span and a multiplication by a non-negative rise each preserve order.
    """

    probe = _calibration_number(value, f"{table.quantity} probe")
    values = table.values
    index = bisect_left(values, probe)
    if index < len(values) and values[index] == probe:
        return table.us[index]
    if index == 0:
        return table.us[0]
    if index == len(values):
        return table.us[-1]

    low_value, high_value = values[index - 1], values[index]
    low_u, high_u = table.us[index - 1], table.us[index]
    reading = low_u + (probe - low_value) / (high_value - low_value) * (high_u - low_u)
    return min(max(reading, low_u), high_u)


# ----------------------------------------------------------------------
# Pool gates: reading the valves, and the two predicates behind them
# ----------------------------------------------------------------------


def _gate_flag(name: str) -> bool:
    """Whether an operator turned this gate on, drain-valve spelling."""

    return os.environ.get(name, "").strip().lower() in _POOL_GATE_ON_FLAGS


def pool_usefulness_floor_from_env() -> float | None:
    """The usefulness floor, or ``None`` when gate (a) is not armed.

    Armed means both halves: ``LM_MAP_POOL_USEFULNESS_GATE`` on *and*
    ``LM_MAP_POOL_MIN_USEFULNESS`` holding a number. Unlike the drain's
    cosine there is no default to fall back to — the value comes off a field
    measurement, and inventing one here would be this module deciding a
    production threshold it has no evidence for. A valve on with no number,
    or with a number that does not parse, therefore leaves the pool exactly as
    it was: the operator asked for a floor and did not say where, and the
    honest answer to that is no floor rather than a guess.
    """

    if not _gate_flag(POOL_USEFULNESS_GATE_ENV):
        return None
    raw = os.environ.get(POOL_USEFULNESS_FLOOR_ENV, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value != value:  # NaN compares false against everything, gate included
        return None
    return value


def pool_demotion_windows_from_env() -> int | None:
    """N in "N known windows without a follow", or ``None`` when unarmed.

    Same two-part rule as :func:`pool_usefulness_floor_from_env`. ``N`` counts
    windows, so it is a positive integer; ``0`` would demote every row with no
    evidence at all and is rejected as the misconfiguration it is rather than
    silently emptying the map.
    """

    if not _gate_flag(POOL_DEMOTION_GATE_ENV):
        return None
    raw = os.environ.get(POOL_DEMOTION_WINDOWS_ENV, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 1 else None


def pool_cold_slots_from_env() -> int | None:
    """The armed cold-lane slot count, or ``None`` when the lane is inert.

    Same paired-valve grammar as the two gates above: the lane runs only when
    :data:`POOL_COLD_QUOTA_GATE_ENV` is on *and*
    :data:`POOL_COLD_SLOTS_ENV` parses to an integer inside the registered
    ``1..POOL_COLD_SLOTS_MAX`` domain. Non-integer, empty, ``0``, negative or
    larger values leave the lane inert rather than guessed at: the registered
    design is a 1-2 slot exploration quota, and a number outside it is not
    this design.
    """

    if not _gate_flag(POOL_COLD_QUOTA_GATE_ENV):
        return None
    raw = os.environ.get(POOL_COLD_SLOTS_ENV, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if 1 <= value <= POOL_COLD_SLOTS_MAX else None


def curtail_decay_from_env() -> bool:
    """Whether the operator armed the decaying curtail retry.

    One flag, no number, unlike the paired gates above: there is no threshold
    to invent because the schedule is the registered rule itself,
    ``min(2**(N-1), CURTAIL_DECAY_SKIP_MAX)``. Read at the two collapse
    decisions — :meth:`RecallMapBuilder.build` and
    :meth:`RecallMapBuilder._with_cold_lane` — and nowhere inside the curtail
    probes, so the unarmed decision path is bit-identical to the frozen
    behaviour and ``test_the_rule_needs_no_flag`` keeps holding.
    """

    return _gate_flag(CURTAIL_DECAY_GATE_ENV)


#: Parse-once memo for :func:`_cold_ranking_tables`, keyed by the identity of
#: the constant so a monkeypatched table set in tests re-parses.
_COLD_TABLES_MEMO: tuple[int, Mapping[str, _CalibrationTable] | None] | None = None


def _cold_ranking_tables() -> Mapping[str, _CalibrationTable] | None:
    """The published cold composite, or ``None`` when the lane cannot rank.

    Fail-closed to inert: a lane whose tables are absent, unreadable under the
    registered form, name an unregistered member, or lack ``result_score``
    delivers nothing rather than ranking on invented numbers. The shipped
    constant is validated by tests; this guard is for the constant a bad
    regeneration could leave behind.
    """

    global _COLD_TABLES_MEMO
    calibration = COLD_RANKING_CALIBRATION
    if _COLD_TABLES_MEMO is not None and _COLD_TABLES_MEMO[0] == id(calibration):
        return _COLD_TABLES_MEMO[1]
    registered = {name for name, _sign in COLD_RANKING_MEMBERS}
    parsed: Mapping[str, _CalibrationTable] | None
    try:
        if (
            not isinstance(calibration, Mapping)
            or "result_score" not in calibration
            or not set(calibration) <= registered
        ):
            parsed = None
        else:
            parsed = {
                name: _calibration_table(calibration, name) for name in calibration
            }
    except ValueError:
        parsed = None
    _COLD_TABLES_MEMO = (id(calibration), parsed)
    return parsed


def _cold_member_values(
    result: Any, tables: Mapping[str, _CalibrationTable]
) -> dict[str, float] | None:
    """The composite members off one residual result, or ``None``.

    Cold eligibility E5: every published member must be readable as a finite
    float off the candidate's own residual record. A candidate with an
    unreadable member is ineligible — ranking it would mean inventing a
    number.
    """

    values: dict[str, float] = {}
    for name in tables:
        try:
            raw = getattr(result, _COLD_MEMBER_FIELDS[name], None)
        except (AttributeError, TypeError, ValueError):
            return None
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        number = float(raw)
        if not isfinite(number):
            return None
        values[name] = number
    return values


def _cold_composite(
    values: Mapping[str, float], tables: Mapping[str, _CalibrationTable]
) -> float:
    """``f = u_rs + u_tg + (1 - u_bm) + (1 - u_vs)`` over the published tables.

    Members are evaluated by :func:`_calibrate` — the merged calibration
    surface — and a negative registered sign contributes ``1 - u``. The
    ordering needs no ``rank_normal`` wrapper: the wrapper is monotone and
    this lane sums no second arm.
    """

    signs = dict(COLD_RANKING_MEMBERS)
    return fsum(
        _calibrate(table, values[name])
        if signs[name] > 0
        else 1.0 - _calibrate(table, values[name])
        for name, table in tables.items()
    )


def _below_usefulness(node: Any, floor: float) -> bool:
    """Whether this candidate's usefulness verdict sits under the floor.

    Read off the node the residual already carries, so no query joins the hot
    path for it. A node whose score is missing or not a real number is *not*
    below the floor: absence of a verdict is not a low verdict, and this gate
    only ever removes rows, so the unknown case has to resolve towards keeping
    them.

    That case is about the *object*, not the corpus. ``nodes.usefulness_score``
    is ``NOT NULL DEFAULT 0.0``, so a node nobody has ever scored arrives here
    as a genuine ``0.0`` and is judged against the floor like any other number
    — which is right, because the floor is chosen from a census of that same
    distribution, zeroes included. The guard is for the duck-typed faces this
    module reads through everywhere else: a result object from another layer
    that has no such attribute must not be silently read as worthless.
    """

    value = getattr(node, "usefulness_score", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    score = float(value)
    if score != score:
        return False
    return score < floor


def _unfollowed_run(history: Any) -> float | None:
    """Softened run of known windows this row was delivered into unfollowed.

    Every input is an aggregate the bounded batched ledger read already
    returned for this candidate (``MemoryStore.matured_recall_history``), so
    the gate adds no query and no per-row work beyond this arithmetic.

    ``lookup_trailing_absent`` is the run itself, and it is already exactly
    what this gate needs: newest-first known windows with no id-fetch,
    stopping at the first followed one, with unobservable (``NULL``) windows
    *skipped* rather than counted — a window that closed before this database
    recorded lookups is not evidence that nobody followed it.

    ``consumed / matured`` is the softening, and it is the row's whole
    re-delivery record rather than a per-window pairing because the ledger
    aggregate is what a batched read can afford. See
    :data:`DEMOTION_REDELIVERY_WEIGHT` for why it may not reach zero.

    ``None`` — unavailable, truncated or self-inconsistent history — means the
    gate abstains. Every other reading of an unreadable history would demote a
    row on the strength of not knowing anything about it.
    """

    if history is None or getattr(history, "available", False) is not True:
        return None
    values = (
        getattr(history, "lookup_trailing_absent", None),
        getattr(history, "lookup_known", None),
        getattr(history, "matured", None),
        getattr(history, "consumed", None),
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        return None
    run, known, matured, consumed = values
    if run < 0 or known < 0 or matured < 0 or consumed < 0:
        return None
    if run > known or known > matured or consumed > matured:
        return None
    if not run:
        return 0.0
    share = consumed / matured if matured else 0.0
    return run * (1.0 - DEMOTION_REDELIVERY_WEIGHT * share)


def _demoted(history: Any, *, ask_followed: bool, after: int) -> bool:
    """Whether gate (b) sinks this row.

    Two exogenous probes, and only exogenous ones. ``ask_followed`` is the
    query-echo probe :meth:`RecallMapBuilder._was_followed` runs, over the
    same window and the same tokenizer, so the two consumers of "did anybody
    follow this" cannot disagree about it; the id-fetch probe arrives inside
    ``history``. The third probe curtail uses — the medoid's ``last_accessed``
    — is deliberately absent, and its absence is the point: it is bumped by
    any recall that returns the node, which is the endogenous echo a hub
    satisfies forever. A gate whose veto is endogenous cannot sink a hub, and
    sinking hubs is the entire job.
    """

    if ask_followed:
        return False
    run = _unfollowed_run(history)
    return run is not None and run >= after


def _selection_gist(node: Any) -> str:
    """An identity-free, compact content example for selection forensics."""

    content = _collapse(getattr(node, "content", ""))
    node_id = getattr(node, "id", None)
    if isinstance(node_id, str) and node_id:
        content = _collapse(content.replace(node_id, ""))
    return _shorten(content, SELECTION_SAMPLE_GIST_CHARS)


def _terms(text: str) -> list[str]:
    """Content terms of ``text``, unstemmed, in order.

    Unstemmed on purpose: these strings are looked up in the FTS
    document-frequency index, whose ``unicode61`` tokenizer does not stem, so a
    stemmed term would report a document frequency of 0 and score as maximally
    rare. Stop-word policy is still the shared one — a raw token is kept when
    ``embeddings.tokenize`` finds anything in it — so this does not fork a
    second stop list.
    """

    kept: list[str] = []
    decided: dict[str, bool] = {}
    for raw in _TERM_RE.findall(str(text).lower()):
        if len(raw) < 3:
            continue
        verdict = decided.get(raw)
        if verdict is None:
            verdict = bool(tokenize(raw))
            decided[raw] = verdict
        if verdict:
            kept.append(raw)
    return kept


def _phrase_from_content(content: str, *, words: int = 3) -> str:
    """A short readable phrase from a node's own content."""

    terms = _terms(content)[:words]
    return " ".join(terms) if terms else _shorten(content, MAX_LABEL_CHARS)


def _followed_hint(stage: str, label: str, previous: str) -> str:
    """The ask-hint a disambiguated label leaves behind.

    On stages 1 and 2 the hint follows the label's *head* rather than the whole
    of it, capped at :data:`ASK_HINT_MAX_TOKENS`. Disambiguation appends a
    distinguishing term to a label, and a hint that followed it there would
    reach three tokens — which is where ``_echoes`` starts demanding two shared
    tokens instead of one and quietly tightens a pre-registered rule.

    The cap is never allowed to *shorten* a hint, and the containment check is
    what makes that a guarantee rather than an intention: the widened hint must
    still carry every token the old one had, under the same tokenizer
    ``_echoes`` uses, or the old hint is kept unchanged. A query that cleared
    the old phrasing therefore clears the new one by construction.

    Stage 4's hint is left following its label exactly as before — a
    c-TF-IDF label is already three tokens, so a fourth changes nothing about
    what ``_echoes`` demands — and stage 3's is never rewritten at all.
    """

    if stage not in (STAGE_PATH, STAGE_STRUCTURAL):
        return label
    keep = max(ASK_HINT_MAX_TOKENS, len(previous.split()))
    candidate = " ".join(label.split()[:keep])
    if not candidate or not token_set(previous) <= token_set(candidate):
        return previous
    return candidate


# ----------------------------------------------------------------------
# Stage helpers
# ----------------------------------------------------------------------


def _structural_key(node: Node) -> tuple[str, str] | None:
    """``(field, raw value)`` of the first structural key this node carries."""

    context = node.context if isinstance(node.context, Mapping) else {}
    for key_field in STRUCTURAL_FIELDS:
        value = context.get(key_field)
        if isinstance(value, str) and value.strip():
            return (key_field, value.strip())
    return None


def _iter_path_strings(node: Node) -> Iterable[str]:
    """Path-like strings this node's context carries.

    Context is read, content is not. A path in prose is a mention; a path in
    ``context['files']`` is a claim about what the node is *about*, which is the
    only thing a subsystem label may be built from. Summarized context values
    (the ``{"count": n, "chars": m}`` shape the delivery diet writes over long
    lists) hold no paths and are skipped by the string check.
    """

    context = node.context if isinstance(node.context, Mapping) else {}
    for path_field in _PATH_FIELDS:
        value = context.get(path_field)
        if isinstance(value, str):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str):
                    yield item


def _subsystem(path: str) -> str | None:
    """Collapse one path to the subsystem that owns it.

    Directory segments only — the file name is the leaf, never the subsystem —
    starting after the last container segment (:data:`_PATH_ROOT_SEGMENTS`) and
    taking the deeper of the first two that remain. That is what makes
    ``src/living_memory/postsession/report.py`` read as ``postsession`` while
    ``src/living_memory/storage.py`` stays ``living_memory``: the rule descends
    exactly as far as the path affords. A path with a container segment and
    nothing after it (``tests/test_map.py``) keeps the container, which for
    ``tests`` and ``docs`` is the right subsystem anyway.
    """

    cleaned = str(path).split(":", 1)[0].strip().replace("\\", "/")
    if "/" not in cleaned:
        return None
    parts = [part for part in cleaned.split("/") if part not in ("", ".", "..")]
    directories = parts[:-1]
    if not directories:
        return None
    start = 0
    for index, segment in enumerate(directories):
        if segment.lower() in _PATH_ROOT_SEGMENTS:
            start = index + 1
    tail = directories[start:]
    if not tail:
        return directories[-1]
    return tail[1] if len(tail) >= 2 else tail[0]


def _node_subsystem(node: Node) -> str | None:
    """The subsystem a node belongs to, when its paths agree on one.

    A node touching several subsystems is assigned to the one it names most
    often, ties broken lexicographically, so the answer does not depend on the
    order a context list happened to be written in.
    """

    counted: Counter[str] = Counter()
    for candidate in _iter_path_strings(node):
        subsystem = _subsystem(candidate)
        if subsystem:
            counted[subsystem] += 1
    if not counted:
        return None
    return min(counted.items(), key=lambda item: (-item[1], item[0]))[0]


def _normalize_vector(values: Sequence[float]) -> list[float] | None:
    norm = sqrt(sum(float(value) * float(value) for value in values))
    if norm <= 0.0:
        return None
    return [float(value) / norm for value in values]


# ----------------------------------------------------------------------
# Consumption: was a delivered map followed?
# ----------------------------------------------------------------------


def _delivered_items(payload: Any) -> list[_DeliveredItem]:
    """The consumable items of one persisted map payload.

    Defensive to the point of paranoia about shape, because this reads a JSON
    blob written by whatever version of this module was running when the
    delivery happened — including versions that do not exist yet. Anything it
    cannot parse contributes no items, which makes that delivery unjudgeable
    rather than unconsumed: a payload this code cannot read is not evidence
    that nobody read it.
    """

    if not isinstance(payload, Mapping):
        return []
    clusters = payload.get("clusters")
    if not isinstance(clusters, (list, tuple)):
        return []
    items: list[_DeliveredItem] = []
    for entry in clusters[:MAX_CLUSTERS]:
        if not isinstance(entry, Mapping):
            continue
        medoid = entry.get("medoid")
        node_id = ""
        if isinstance(medoid, Mapping):
            node_id = str(medoid.get("node_id") or "")
        items.append(
            _DeliveredItem(
                medoid_id=node_id,
                label=token_set(str(entry.get("label") or "")),
                ask_hint=token_set(str(entry.get("ask_hint") or "")),
            )
        )
    return items


def _echoes(phrasing: frozenset[str], query: frozenset[str]) -> bool:
    """Whether ``query`` carries enough of ``phrasing`` to count as asking it.

    Containment of the *phrasing* in the query, not the reverse: a plan item
    says "recall X", and the recall that follows it is free to say more than X
    — it usually does. Weighting the shared tokens by rarity, as
    :mod:`living_memory.grounding` does for content, would need a corpus this
    path has no budget to build; a label is a handful of tokens the cascade
    already chose for being distinguishing, so unweighted containment over the
    shared tokenizer is the honest measure at this size.
    """

    if not phrasing or not query:
        return False
    shared = len(phrasing & query)
    return shared >= CURTAIL_QUERY_OVERLAP * len(phrasing)


# ----------------------------------------------------------------------
# Cheap revision probes, mirroring retrieval's
# ----------------------------------------------------------------------


def _data_version(connection: sqlite3.Connection) -> int:
    """``PRAGMA data_version``: bumped when another connection commits."""

    row = connection.execute("PRAGMA data_version").fetchone()
    return int(row[0]) if row is not None else 0


def _chunk_table_revision(connection: sqlite3.Connection) -> tuple[Any, ...]:
    """``(row count, greatest chunk id)``, or ``()`` on a pre-v6 database.

    The same two statements, and the same completeness argument, as
    ``retrieval._chunk_table_revision``: chunk rows are only inserted or
    deleted, never updated in place, so a delete moves the count and an insert
    raises the ULID maximum. Kept here rather than imported because importing
    ``retrieval`` would pull the whole ranker — and its embedding model — into
    a module that exists to stay cheap; ``tests/test_recall_map.py`` pins the
    two against each other.
    """

    try:
        counted = connection.execute(
            f"SELECT COUNT(*) FROM {CHUNK_EMBEDDING_TABLE}"
        ).fetchone()
        greatest = connection.execute(
            f"SELECT MAX(id) FROM {CHUNK_EMBEDDING_TABLE}"
        ).fetchone()
    except sqlite3.OperationalError:
        return ()
    return (int(counted[0]), greatest[0])


# ----------------------------------------------------------------------
# The builder
# ----------------------------------------------------------------------


class RecallMapBuilder:
    """Builds recall maps, and remembers their shape between calls.

    One instance per service, like ``MemoryRecallService``'s own caches: the
    stability guarantee is "same task, same structure", which only means
    anything if the thing holding it outlives a single call.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        max_clusters: int = MAX_CLUSTERS,
        max_pool_nodes: int = MAX_POOL_NODES,
        cluster_cosine: float = EMBEDDING_CLUSTER_COSINE,
    ) -> None:
        self.store = store
        self.max_clusters = max(1, int(max_clusters))
        self.max_pool_nodes = max(1, int(max_pool_nodes))
        self.cluster_cosine = float(cluster_cosine)
        #: Whether the last :meth:`build` served its structure from cache.
        #: Read by the latency gate and by the stability tests.
        self.last_cache_hit: bool = False
        #: What the last :meth:`build` found about this key's unused streak,
        #: including the builds that did *not* collapse — the response only
        #: carries the number once it has become a decision, and the runs up to
        #: that point are what a latency gate or a test needs to see.
        self.last_curtailment: _Curtailment = _Curtailment(streak=0, offers=0)
        #: Whether the last :meth:`build` paid for a history read, or answered
        #: the curtail question from what it had already delivered itself.
        self.last_curtail_read: bool = False
        #: Medoids the last curtail question found ask-echoed under that key,
        #: empty unless the demotion valve asked for them. Exposed the way the
        #: three above are: a by-product of a read the build already paid for,
        #: which the pool gate consumes and tests can inspect.
        self.last_ask_follows: frozenset[str] = frozenset()
        self._cache: dict[str, _CachedStructure] = {}
        self._curtail_memo: dict[str, _CurtailMemo] = {}
        self._write_probe: tuple[int, int] | None = None
        self._revision: tuple[Any, ...] | None = None
        #: Corpus statistics for the build in flight, reset by :meth:`build`.
        #: Label enrichment asks the document-frequency index once per bucket
        #: and the gate once per label, which without a memo would be one
        #: ``COUNT(*)`` per question; the corpus cannot move under a single
        #: build, so one read answers all of them.
        self._fts_total: int | None = None
        self._term_df: dict[str, int] = {}
        self._df_available = True

    # -- public API ----------------------------------------------------

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def build(
        self,
        results: Sequence["RecallResult"],
        *,
        scope: str | None = None,
        task: str | None = None,
        task_pattern: str | None = None,
        decision_at: str | datetime | None = None,
    ) -> RecallMap | None:
        """Cluster one residual pool. ``None`` when there is nothing to say.

        ``results`` is the ranked residual — ``MemoryRecallService.last_residual``
        — best first. Its original order supplies the frozen ordinal after
        relevance selection; the selected pool itself is ordered by score.
        ``task``/``task_pattern`` name the work the recall belongs to and, with
        ``scope``, form the cache key; without either, the key degrades to the
        scope alone, which is still stable and merely coarser.

        ``decision_at`` is captured once per call so every bounded history
        batch sees one boundary. Supplying it is useful for deterministic
        replay; production callers omit it and receive the current UTC instant.

        A key whose last :data:`CURTAIL_STREAK` maps were delivered and never
        followed gets the collapsed form instead — clusters empty, ``curtailed``
        set, the streak carried — and gets it *before* any clustering runs, so
        a channel nobody reads also stops costing what it costs to fill. With
        :data:`CURTAIL_DECAY_GATE_ENV` armed a collapsed key that has served
        its skip run (:attr:`_Curtailment.retry_due`) rebuilds its pool and,
        only if the pool moved since the key last looked (:meth:`_reoffer`),
        builds a full map again instead of the marker; a map that comes back
        with nothing deliverable in it is persisted as a marker too, carrying
        the new baseline. Nothing else on either path moves.

        With :data:`POOL_DEMOTION_GATE_ENV` armed the curtail verdict is read
        *before* the pool instead of after it, because the gate needs the
        ask-echo probe that read already computes. Nothing else moves: the
        read is the same read behind the same memo, the verdict is carried
        forward rather than asked for twice, and with the valve unset the
        order below is untouched.
        """

        self.last_cache_hit = False
        self._reset_corpus_memo()
        if not results:
            return None

        instant = decision_at if decision_at is not None else datetime.now(UTC)
        # Hoisted above `_pool` so the gate can be handed this key's evidence.
        # Pure string work on arguments already in hand — no read, no cache
        # touch — so an unarmed build computes exactly what it always did, in
        # a different order that nothing downstream can observe.
        map_scope = (
            normalize_scope(scope) if scope else _dominant_residual_scope(results)
        )
        key = cache_key(map_scope, task=task, task_pattern=task_pattern)
        demote_after = pool_demotion_windows_from_env()
        cold_slots = pool_cold_slots_from_env()
        # The capture list exists only when the pair is armed, so the unarmed
        # `_pool` call carries `None` and the capture branch never runs.
        cold_rows: list[tuple[int, Node, Any, Any]] | None = (
            [] if cold_slots is not None else None
        )
        curtailment: _Curtailment | None = None
        ask_followed: frozenset[str] = frozenset()
        if demote_after is not None:
            # One consequence of asking early is worth naming rather than
            # discovering: a build whose pool comes back empty now seeds a
            # curtail memo where it previously seeded none, so `_note_delivery`
            # counts it. That is the honest count — the build did deliver, and
            # it did offer nothing — and it cannot cause a collapse, because a
            # collapse is decided by `offers`, which an empty pool never
            # advances. It can only make the reported streak estimate less of
            # an undercount.
            curtailment = self._curtailment(
                map_scope, task, task_pattern, want_ask_follows=True
            )
            ask_followed = self.last_ask_follows
        pool, selection = self._pool(
            results,
            decision_at=instant,
            demote_after=demote_after,
            ask_followed=ask_followed,
            cold_rows=cold_rows,
        )
        if not pool:
            # A non-empty residual that policy filtered completely is evidence,
            # not the same event as an empty residual.  It offered no cluster
            # and therefore cannot advance curtailment.  Its empty structure
            # is still a useful negative cache result, but only for a fresh
            # empty selection under this exact corpus/policy revision.  Keep
            # this branch explicit: `_recount((), pool)` deliberately rejects
            # so an empty entry can never hide a later eligible pool.
            revision = self._corpus_revision()
            cached = self._cache.get(key)
            if (
                cached is not None
                and cached.revision == revision
                and not cached.templates
            ):
                self.last_cache_hit = True
            else:
                self._remember(key, revision, [])
            self.last_curtailment = _Curtailment(streak=0, offers=0)
            built = RecallMap(
                scope=map_scope,
                key=key,
                clusters=(),
                pool_size=0,
                covered=0,
                selection=selection,
                selection_sample_limit=_selection_sample_limit(
                    (), 0, 0, selection=selection
                ),
            )
            if cold_slots is not None:
                # The one path that never computed a curtail verdict, so the
                # lane asks for it itself before delivering anything.
                built = self._with_cold_lane(
                    built,
                    cold_rows or (),
                    slots=cold_slots,
                    scope=map_scope,
                    task=task,
                    task_pattern=task_pattern,
                    read_curtailment=True,
                )
            self._note_delivery(
                map_scope, task, task_pattern, offered=bool(built.clusters)
            )
            return built

        if curtailment is None:
            curtailment = self._curtailment(map_scope, task, task_pattern)
        self.last_curtailment = curtailment

        # The lane's eligible candidates, computed at most once per build: the
        # pool digest needs them on a collapsed key, the lane needs them when
        # it runs, and the probe behind them is a read.
        cold_candidates: list[tuple[int, Node, Any, dict[str, float]]] | None = None

        def digest() -> str:
            nonlocal cold_candidates
            if cold_rows is not None and cold_candidates is None:
                cold_candidates = self._cold_candidates(cold_rows)
            return self._pool_digest(pool, cold_rows, candidates=cold_candidates)

        # The decay valve can only turn a collapse into a retry, never the
        # reverse, so an unarmed build takes exactly the frozen branch and
        # never computes a digest. A retry that delivers is a plain map: the
        # next read sees it as the newest offer, zero lead, and the skip run
        # starts over — doubled, if it too goes unconsumed.
        offer, pool_digest = self._reoffer(curtailment, digest)
        if not offer:
            self._note_delivery(map_scope, task, task_pattern, offered=False)
            return self._collapsed(
                map_scope,
                key,
                pool_size=len(pool),
                streak=curtailment.streak,
                selection=selection,
                # On a curtailed key the lane does not run: the unread-collapse
                # defense binds cold content exactly as warm, and an armed
                # server states that fact on the wire as c = [0, 0].
                cold=(0, 0) if cold_slots is not None else None,
                pool_digest=pool_digest,
            )

        revision = self._corpus_revision()
        cached = self._cache.get(key)
        groups: list[_Group] | None = None
        if (
            cached is not None
            and cached.revision == revision
            and cached.templates
        ):
            groups = self._recount(cached.templates, pool)
            self.last_cache_hit = groups is not None
        if groups is None:
            groups = self._cluster(pool)
            self._remember(key, revision, groups)
        built = self._finish(
            groups,
            scope=map_scope,
            key=key,
            pool_size=len(pool),
            selection=selection,
        )
        if built is not None and cold_slots is not None:
            # The curtail verdict on this path was already read and either was
            # not a collapse or was reprieved — the collapse branch returned
            # above — so the lane runs against the finished warm map.
            built = self._with_cold_lane(
                built,
                cold_rows or (),
                slots=cold_slots,
                scope=map_scope,
                task=task,
                task_pattern=task_pattern,
                read_curtailment=False,
                candidates=cold_candidates,
            )
        if built is not None and pool_digest is not None and not built.clusters:
            # A reprieved key that rebuilt its map and found nothing deliverable
            # in it did not wake up. The wire says so — a marker, not an empty
            # uncollapsed map — and the marker carries the digest of the pool
            # that had nothing to say, so the next look compares against it
            # rather than re-discovering the same nothing every delivery.
            self._note_delivery(map_scope, task, task_pattern, offered=False)
            return self._collapsed(
                map_scope,
                key,
                pool_size=len(pool),
                streak=curtailment.streak,
                selection=selection,
                cold=built.cold,
                pool_digest=pool_digest,
            )
        if built is not None:
            # A journal-only map reaches the server so a fully gated pool does
            # not disappear from field evidence, but it offered the agent no
            # cluster and therefore must not advance the curtail offer count.
            self._note_delivery(
                map_scope, task, task_pattern, offered=bool(built.clusters)
            )
        return built

    # -- pool ----------------------------------------------------------

    def _pool(
        self,
        results: Sequence["RecallResult"],
        *,
        decision_at: str | datetime,
        demote_after: int | None = None,
        ask_followed: Collection[str] = (),
        cold_rows: list[tuple[int, Node, Any, Any]] | None = None,
    ) -> tuple[list[_Member], SelectionAccounting]:
        """Apply the frozen member pipeline before the pool cap.

        Classification is one ordinal pass.  History is then read only for
        first-occurrence, structurally eligible identities, in batches no
        larger than storage's frozen bound.  Every survivor is scored before
        the stable relevance sort and cap, so machine ballast and low-score
        head entries cannot crowd a useful tail member out of the 200.

        Two env-gated gates sit *around* that pipeline and never inside it.
        Gate (a) — a ``usefulness_score`` floor — runs with the ballast rules,
        before history is read, because a candidate the floor rejects should
        not cost a ledger row.  Gate (b) — the unfollowed-window demotion —
        runs after the frozen threshold has spoken, so ``nf`` counts exactly
        the rows the gate newly removed rather than re-labelling rows ``lr``
        would have dropped anyway.  Both read only what is already in hand:
        the score off the node the residual carries, the window aggregates off
        the same bounded batch :meth:`_matured_history` already fetches, and
        ``ask_followed`` off the curtail read the build did before calling
        here.  Neither adds a query, and neither adds a per-row one.

        The two valves are read in different places, and the asymmetry is
        deliberate. Gate (a) needs nothing but this method's own arguments, so
        it reads its valve here. Gate (b) needs the ask-echo evidence, which
        only :meth:`build` can obtain and only when it knows in advance that
        the gate is armed — so ``demote_after`` arrives as an argument, and a
        caller that reaches ``_pool`` directly gets no demotion rather than a
        demotion decided without the evidence that holds rows back.

        With both valves unset ``demote_after`` is ``None``, the floor is
        ``None``, nothing below branches, and the ledger trims back to the
        frozen seven-code vector.

        What survives classification is the ``RecallResult`` and not just its
        node. The result is where a node's *own* evidence for this query lives
        — see :data:`_CANDIDATE_SCORE_FIELDS` — and a node with no delivery
        history owns nothing else; this pass is the last place that evidence
        exists, so dropping it here is what would make a cold-start scorer
        unbuildable. :func:`relevance_score` takes it and, under the frozen
        policy, declines it: carrying it moves no score.
        """

        usefulness_floor = pool_usefulness_floor_from_env()
        ledger = _SelectionLedger(inspected=len(results))
        seen: set[str] = set()
        eligible: list[tuple[int, Node, "RecallResult"]] = []
        for ordinal, result in enumerate(results):
            node = getattr(result, "node", None)
            node_id = getattr(node, "id", None) if node is not None else None
            if not isinstance(node_id, str) or not node_id.strip():
                ledger.exclude("iv", ordinal=ordinal)
                continue
            if node_id in seen:
                ledger.exclude("du", ordinal=ordinal)
                continue
            seen.add(node_id)

            reason = _ballast_reason(
                getattr(node, "content", None),
                getattr(node, "context", None),
                getattr(node, "provenance", None),
            )
            if reason is not None:
                ledger.exclude(reason, ordinal=ordinal, node=node)
                continue
            if usefulness_floor is not None and _below_usefulness(
                node, usefulness_floor
            ):
                ledger.exclude("uf", ordinal=ordinal, node=node)
                continue
            eligible.append((ordinal, node, result))

        histories = self._matured_history(
            [node.id for _ordinal, node, _result in eligible],
            decision_at=decision_at,
        )
        survivors: list[_Member] = []
        for ordinal, node, result in eligible:
            history = histories.get(node.id)
            score = relevance_score(node, history, result)
            if score < RELEVANCE_THRESHOLD:
                ledger.exclude("lr", ordinal=ordinal, node=node)
                # Cold-lane candidate capture: rows the frozen threshold alone
                # excluded, having survived every check above it — eligibility
                # E1 and E2 of the registered plan, by construction rather than
                # by a mirror that could drift. Capture only; classification,
                # scoring and admission are untouched, and with the valve pair
                # unset ``cold_rows`` is ``None`` and this line never runs.
                if cold_rows is not None:
                    cold_rows.append((ordinal, node, result, history))
                continue
            if demote_after is not None and _demoted(
                history,
                ask_followed=node.id in ask_followed,
                after=demote_after,
            ):
                ledger.exclude("nf", ordinal=ordinal, node=node)
                continue
            survivors.append(
                _Member(rank=ordinal, node=node, relevance_score=score)
            )

        survivors.sort(key=lambda member: (-member.relevance_score, member.rank))
        admitted = survivors[: self.max_pool_nodes]
        for member in sorted(
            survivors[self.max_pool_nodes :], key=lambda item: item.rank
        ):
            ledger.exclude("pc", ordinal=member.rank, node=member.node)
        accounting = ledger.freeze(len(admitted))
        return admitted, accounting

    def _matured_history(
        self,
        node_ids: Sequence[str],
        *,
        decision_at: str | datetime,
    ) -> dict[str, Any]:
        """Read every candidate's history through fixed-size bounded batches.

        A missing method, failed batch, absent id, malformed row, truncation or
        unavailable ledger remains absent in this mapping (or carries the
        reader's unavailable object) and therefore receives frozen train means
        in :func:`relevance_score`.  A real zero-history row is present with
        ``available=True`` and M=C=K=0, preserving the cold-start distinction.
        """

        histories: dict[str, Any] = {}
        for offset in range(0, len(node_ids), MAX_RECALL_HISTORY_CANDIDATES):
            batch = node_ids[offset : offset + MAX_RECALL_HISTORY_CANDIDATES]
            try:
                fetched = self.store.matured_recall_history(batch, decision_at)
            except (AttributeError, sqlite3.Error, TypeError, ValueError):
                continue
            if not isinstance(fetched, Mapping):
                continue
            for node_id in batch:
                if node_id in fetched:
                    histories[node_id] = fetched[node_id]
        return histories

    # -- cold exploration lane -----------------------------------------

    def _cold_undelivered(self, node_ids: Sequence[str]) -> frozenset[str] | None:
        """Ids with affirmatively zero delivery-ledger rows, or ``None``.

        Cold eligibility E3 tests row *existence*, not maturity:
        ``matured_recall_history`` cannot express "never delivered", because a
        node delivered an hour ago has ``matured == 0`` and is
        indistinguishable from a virgin node on the matured aggregate — a lane
        keyed on that aggregate would re-deliver the same top-ranked cold node
        on every map for the 24h its first window takes to mature.

        E4 makes the unknown case fail towards not-delivering: an absent or
        incomplete ledger, a wrong format, or a failed read returns ``None``
        and every candidate becomes ineligible — the mirror of
        :func:`_below_usefulness`'s fail-open, inverted because this lane only
        ever *adds* delivery. The state checks are the same three
        :meth:`MemoryStore.matured_recall_history` makes before trusting the
        ledger.
        """

        ids = list(dict.fromkeys(str(node_id) for node_id in node_ids))
        if not ids:
            return frozenset()
        try:
            state = self.store.connection.execute(
                f"""
                SELECT format_version, complete
                FROM {RECALL_DELIVERY_HISTORY_STATE_TABLE}
                WHERE singleton = 1
                """
            ).fetchone()
            if (
                state is None
                or int(state["format_version"]) != _RECALL_DELIVERY_HISTORY_FORMAT
                or not int(state["complete"])
            ):
                return None
            delivered: set[str] = set()
            for offset in range(0, len(ids), MAX_RECALL_HISTORY_CANDIDATES):
                batch = ids[offset : offset + MAX_RECALL_HISTORY_CANDIDATES]
                placeholders = ",".join("?" for _ in batch)
                rows = self.store.connection.execute(
                    f"""
                    SELECT DISTINCT node_id
                    FROM {RECALL_DELIVERY_HISTORY_TABLE}
                    WHERE node_id IN ({placeholders})
                    """,
                    batch,
                ).fetchall()
                delivered.update(str(row["node_id"]) for row in rows)
        except (AttributeError, sqlite3.Error, TypeError, ValueError):
            return None
        return frozenset(ids) - delivered

    def _cold_candidates(
        self, cold_rows: Sequence[tuple[int, Node, Any, Any]]
    ) -> list[tuple[int, Node, Any, dict[str, float]]]:
        """The lane's eligible candidates: E1-E5 survivors, before ranking.

        Captured rows whose composite the published tables can value, that
        carry no matured window, and that the delivery ledger affirmatively
        has never delivered. Empty — not ``None`` — when the lane cannot rank
        or the ledger cannot answer, because on both counts the lane delivers
        nothing and a digest of "nothing eligible" is the honest one. One
        ledger probe per call; :meth:`build` calls it at most once per build
        and hands the result to both the digest and the lane.
        """

        tables = _cold_ranking_tables()
        if tables is None or not cold_rows:
            return []

        candidates: list[tuple[int, Node, Any, dict[str, float]]] = []
        probe_ids: list[str] = []
        for ordinal, node, result, history in cold_rows:
            values = _cold_member_values(result, tables)
            if values is None:
                continue
            matured = (
                getattr(history, "matured", None)
                if history is not None and getattr(history, "available", False)
                else None
            )
            if matured is not None and matured > 0:
                # A matured window is a ledger row; no existence probe needed.
                continue
            candidates.append((ordinal, node, result, values))
            probe_ids.append(node.id)

        undelivered = self._cold_undelivered(probe_ids)
        if undelivered is None:
            return []
        return [
            (ordinal, node, result, values)
            for ordinal, node, result, values in candidates
            if node.id in undelivered
        ]

    def _cold_lane(
        self,
        cold_rows: Sequence[tuple[int, Node, Any, Any]],
        *,
        slots: int,
        warm_clusters: Sequence[MapCluster],
        candidates: Sequence[tuple[int, Node, Any, dict[str, float]]] | None = None,
    ) -> tuple[int, list[MapCluster]]:
        """Rank the captured cold candidates and deliver into free capacity.

        Returns ``(g, delivered)``: the count of cold-eligible candidates
        examined this build (E1-E5 survivors, before ranking and guards) and
        the marked singleton clusters to append. Candidates are ranked only
        against each other on the published composite; the tie-break — higher
        raw ``result_score``, then lower residual ordinal — makes the order
        total and deterministic. Two per-candidate guards run at delivery
        time, each skipping to the next-ranked candidate: the same
        deliverable-label gate every warm cluster passes, and the inter-slot
        near-duplicate guard at :data:`EMBEDDING_CLUSTER_COSINE`, which
        abstains (delivers) when either embedding is unavailable — absence of
        a verdict is not a duplicate verdict.

        ``candidates`` is the eligible list a caller already computed through
        :meth:`_cold_candidates`; absent, the lane computes it here.
        """

        eligible = (
            self._cold_candidates(cold_rows) if candidates is None else candidates
        )
        examined = len(eligible)
        tables = _cold_ranking_tables()
        capacity = min(int(slots), self.max_clusters - len(warm_clusters))
        if tables is None or capacity <= 0 or not eligible:
            return examined, []

        def order(entry: tuple[int, Node, Any, dict[str, float]]) -> tuple:
            ordinal, _node, result, values = entry
            raw = getattr(result, "score", None)
            raw_score = (
                float(raw)
                if isinstance(raw, (int, float)) and not isinstance(raw, bool)
                else 0.0
            )
            return (-_cold_composite(values, tables), -raw_score, ordinal)

        delivered: list[tuple[MapCluster, list[float] | None]] = []
        for ordinal, node, _result, _values in sorted(eligible, key=order):
            if len(delivered) >= capacity:
                break
            member = _Member(rank=ordinal, node=node, relevance_score=0.0)
            label = self._ctfidf_label([member])
            if not self._deliverable(label):
                continue
            vector = self._node_vector(node)
            duplicate = any(
                prior is not None
                and vector is not None
                and len(prior) == len(vector)
                and cosine_similarity(prior, vector) >= self.cluster_cosine
                for _cluster, prior in delivered
            )
            if duplicate:
                continue
            group = _Group(
                stage=STAGE_COLD,
                signature=(STAGE_COLD, (node.id,)),
                label=label,
                ask_hint="",
                members=[member],
            )
            cluster = replace(
                self._cluster_of(group, example_chars=MEDOID_EXAMPLE_CHARS),
                cold=True,
            )
            delivered.append((cluster, vector))
        return examined, [cluster for cluster, _vector in delivered]

    def _with_cold_lane(
        self,
        built: RecallMap,
        cold_rows: Sequence[tuple[int, Node, Any, Any]],
        *,
        slots: int,
        scope: str,
        task: str | None,
        task_pattern: str | None,
        read_curtailment: bool,
        candidates: list[tuple[int, Node, Any, dict[str, float]]] | None = None,
    ) -> RecallMap:
        """Append the lane's outcome to a built map, warm bytes untouched.

        The warm cluster list, counts, ``sel`` core and sample budget were all
        settled before this runs, so with the valves armed the only changes
        are the appended marked clusters and the ``c`` key — the additive
        contract in its constructive form. ``read_curtailment`` asks for the
        unread-collapse verdict on the one path that never computed it (the
        empty-pool build): on a key the machinery would collapse, the lane
        does not run and an armed build reports ``c = [0, 0]``, because the
        collapse defense binds cold content exactly as warm — including the
        decay reprieve, which reopens the lane on the same delivery it would
        reopen a warm map, and the pool-aware look that precedes it: on this
        path the pool is the lane's eligible candidates, and a served run that
        finds them unchanged states ``c = [0, 0]`` and the digest, delivering
        nothing. The wire on this path never carries ``curtailed`` — the
        collapse machinery issued no such verdict for an empty pool — so the
        digest rides the empty map exactly as it rides a marker.

        Appended cold examples are shrunk — cold examples only, never a warm
        byte — towards :data:`MAX_RESPONSE_CHARS`; if even bare clusters
        overflow, the lane still delivers, because the registered noise bound
        is the slot quota, not the character budget the warm map was fitted
        under.
        """

        pool_digest = built.pool_digest
        if read_curtailment:
            verdict = self._curtailment(scope, task, task_pattern)

            def digest() -> str:
                nonlocal candidates
                if candidates is None:
                    candidates = self._cold_candidates(cold_rows)
                return self._pool_digest((), cold_rows, candidates=candidates)

            offer, pool_digest = self._reoffer(verdict, digest)
            if not offer:
                return replace(built, cold=(0, 0), pool_digest=pool_digest)
        examined, cold_clusters = self._cold_lane(
            cold_rows,
            slots=slots,
            warm_clusters=built.clusters,
            candidates=candidates,
        )
        if not cold_clusters:
            return replace(built, cold=(examined, 0), pool_digest=pool_digest)

        outcome = (examined, len(cold_clusters))
        filtered = _filter_block(
            built.withheld, built.dropped, built.filtered, built.filtered_omitted
        )

        def size_at(width: int) -> tuple[int, tuple[MapCluster, ...]]:
            shaped = tuple(
                replace(
                    cluster,
                    medoid=replace(
                        cluster.medoid,
                        example=_shorten(cluster.medoid.example, width),
                    ),
                )
                for cluster in cold_clusters
            )
            return (
                _payload_size(
                    (*built.clusters, *shaped),
                    built.pool_size,
                    built.dropped,
                    filtered,
                    selection=built.selection,
                    selection_sample_limit=built.selection_sample_limit,
                    cold=outcome,
                ),
                shaped,
            )

        low, high = 0, MEDOID_EXAMPLE_CHARS
        _size, best = size_at(0)
        while low <= high:
            width = (low + high) // 2
            measured, shaped = size_at(width)
            if measured <= MAX_RESPONSE_CHARS:
                best = shaped
                low = width + 1
            else:
                high = width - 1
        return replace(
            built,
            clusters=(*built.clusters, *best),
            covered=built.covered + sum(cluster.count for cluster in best),
            cold=outcome,
        )

    # -- cascade -------------------------------------------------------

    def _cluster(self, pool: list[_Member]) -> list[_Group]:
        """Run the four stages, each over what the previous one left."""

        groups: list[_Group] = []
        remaining = pool

        claimed, remaining = self._stage_structural(remaining)
        groups.extend(claimed)
        claimed, remaining = self._stage_path(remaining)
        groups.extend(claimed)
        claimed, remaining = self._stage_anchor(remaining)
        groups.extend(claimed)
        claimed, remaining = self._stage_embedding(remaining)
        groups.extend(claimed)
        return self._disambiguate(groups)

    def _disambiguate(self, groups: list[_Group]) -> list[_Group]:
        """Make every label unique, because the map is a menu.

        Two rows reading the same text are a defect whatever produced them —
        the reader cannot act differently on them, and a repeated label is the
        degenerate map this cascade exists to avoid. Collisions are not
        hypothetical: two c-TF-IDF clusters can share their top terms, and
        ``type: root-cause`` and ``lesson_kind: root_cause`` normalize to the
        same words from two different fields.

        The first group to claim a label keeps it; later ones extend theirs with
        their own next-most-distinguishing term, and fall back to a numeric
        suffix only if a group has no term left to offer. Merging the
        colliding groups instead would be simpler and wrong: they were
        separated by evidence — different keys, different vector regions — and
        a label collision is a failure to *name* that difference, not proof
        that it is absent.

        Uniqueness is enforced on the *displayed* label, already cut to
        :data:`MAX_LABEL_CHARS`. Deciding it on the full string instead would
        let two anchor queries sharing a long prefix — the house style that
        makes anchors work in the first place — pass this pass and collide
        again in the response.
        """

        seen: set[str] = set()
        for group in groups:
            # An anchor's ask-hint is a query that already worked, and it is
            # never rewritten — not even though it starts out equal to the
            # label. Everywhere else a hint that *was* the label follows it,
            # but only as far as :func:`_followed_hint` allows: on the enriched
            # stages the hint tracks the label's head, not its whole width.
            hint_followed_label = (
                group.stage != STAGE_ANCHOR and group.ask_hint == group.label
            )
            resolved = self._unique_label(group, seen)
            seen.add(resolved)
            if hint_followed_label:
                group.ask_hint = _followed_hint(group.stage, resolved, group.ask_hint)
            group.label = resolved
        return groups

    def _unique_label(self, group: _Group, seen: set[str]) -> str:
        label = _shorten(group.label, MAX_LABEL_CHARS)
        if label and label not in seen:
            return label
        for term in self._ranked_terms(group.members):
            if term in label.split():
                continue
            # Make room for the term rather than overflowing the width, so the
            # distinguishing part cannot be the part that gets cut off.
            base = _shorten(group.label, max(1, MAX_LABEL_CHARS - len(term) - 1))
            candidate = f"{base} {term}"
            if candidate not in seen:
                return candidate
        suffix = 2
        while True:
            marker = f" #{suffix}"
            candidate = f"{_shorten(group.label, MAX_LABEL_CHARS - len(marker))}{marker}"
            if candidate not in seen:
                return candidate
            suffix += 1

    def _stage_structural(
        self, members: list[_Member]
    ) -> tuple[list[_Group], list[_Member]]:
        buckets: dict[tuple[str, str], list[_Member]] = {}
        rest: list[_Member] = []
        for member in members:
            key = _structural_key(member.node)
            if key is None:
                rest.append(member)
                continue
            buckets.setdefault(key, []).append(member)

        groups: list[_Group] = []
        for (structural_field, raw), bucket in sorted(buckets.items()):
            normalized = normalize_key(raw)
            if _looks_unreadable(raw):
                # A hash groups perfectly and names nothing; the medoid does.
                readable = _phrase_from_content(bucket[0].node.content)
            else:
                readable = normalized
            label, hint = readable, readable
            if not self._deliverable(readable):
                # A structural value that is one house word is exactly as
                # undeliverable as a bare directory segment — 6 of the 55
                # ``server`` clusters in the field window were structural, not
                # path — so it gets the same rescue. A value that already
                # passes is left alone: ``cleanup commit invariant`` needs no
                # help, and rewriting a key the corpus actually stores would
                # throw away the one label form the control run proved works.
                label, hint = self._enrich(readable, bucket)
            groups.append(
                _Group(
                    stage=STAGE_STRUCTURAL,
                    signature=(STAGE_STRUCTURAL, structural_field, raw),
                    label=label,
                    ask_hint=hint,
                    members=bucket,
                )
            )
        return groups, rest

    def _stage_path(self, members: list[_Member]) -> tuple[list[_Group], list[_Member]]:
        """Collapse paths to subsystems, then name the subsystem's own content.

        The subsystem alone is a directory, and a directory is what the field
        measured as the dominant failure of this map: enrichment is
        unconditional here rather than gated on the score, because a path
        segment is never the cluster's own vocabulary — it is where the files
        happen to live.
        """

        buckets: dict[str, list[_Member]] = {}
        rest: list[_Member] = []
        for member in members:
            subsystem = _node_subsystem(member.node)
            if subsystem is None:
                rest.append(member)
                continue
            buckets.setdefault(subsystem, []).append(member)

        groups: list[_Group] = []
        for subsystem, bucket in sorted(buckets.items()):
            label, hint = self._enrich(normalize_key(subsystem), bucket)
            groups.append(
                _Group(
                    stage=STAGE_PATH,
                    signature=(STAGE_PATH, subsystem),
                    label=label,
                    ask_hint=hint,
                    members=bucket,
                )
            )
        return groups, rest

    def _anchor_assignment(
        self, members: Sequence[_Member]
    ) -> tuple[dict[str, str], dict[str, Any], list[_Member]]:
        """``node_id -> anchor_id`` for every member a live anchor covers.

        Also returns the anchors reached, by id, and the members no live
        anchor covers, in their input order. This is stage 3's partition
        input, factored out so the pool digest can name the anchors that
        would take part without running the stage.
        """

        anchors: dict[str, Any] = {}
        assignment: dict[str, str] = {}
        rest: list[_Member] = []
        for member in members:
            best: tuple[float, str] | None = None
            for edge in self.store.list_query_anchor_edges(target_id=member.node.id):
                anchor = anchors.get(edge.anchor_id)
                if anchor is None:
                    anchor = self.store.get_query_anchor(edge.anchor_id)
                    if anchor is None:
                        continue
                    anchors[edge.anchor_id] = anchor
                if anchor.decayed or not str(anchor.query).strip():
                    continue
                weight = float(edge.weight)
                if best is None or (-weight, edge.anchor_id) < (-best[0], best[1]):
                    best = (weight, edge.anchor_id)
            if best is None:
                rest.append(member)
            else:
                assignment[member.node.id] = best[1]
        return assignment, anchors, rest

    def _stage_anchor(
        self, members: list[_Member]
    ) -> tuple[list[_Group], list[_Member]]:
        """Group by the live query anchor whose edges cover each node.

        One indexed lookup per unclaimed node — ``idx_query_anchor_edges_target``
        exists precisely for this direction — and one anchor row per distinct
        anchor reached, memoized. A node covered by several anchors goes to the
        heaviest edge, ties to the lower anchor id, so the assignment does not
        depend on the order the edge table returns.
        """

        assignment, anchors, rest = self._anchor_assignment(members)

        buckets: dict[str, list[_Member]] = {}
        for member in members:
            anchor_id = assignment.get(member.node.id)
            if anchor_id is not None:
                buckets.setdefault(anchor_id, []).append(member)

        groups: list[_Group] = []
        for anchor_id, bucket in sorted(buckets.items()):
            query = _collapse(anchors[anchor_id].query)
            groups.append(
                _Group(
                    stage=STAGE_ANCHOR,
                    signature=(STAGE_ANCHOR, anchor_id),
                    label=query,
                    # The anchor's own wording, unedited: it is not a guess at
                    # how to ask, it is a question that already worked.
                    ask_hint=query,
                    members=bucket,
                )
            )
        return groups, rest

    def _stage_embedding(
        self, members: list[_Member]
    ) -> tuple[list[_Group], list[_Member]]:
        """Greedy leader clustering over chunk vectors, c-TF-IDF labels.

        Leader rather than centroid clustering, because a moving centroid makes
        membership depend on arrival order in a way the rank order alone cannot
        justify: here a node joins the *first-seeded* cluster it is closest to,
        and the seed is by construction the highest-ranked member of it.
        """

        vectors: dict[str, list[float]] = {}
        clusterable: list[_Member] = []
        rest: list[_Member] = []
        for member in members:
            vector = self._node_vector(member.node)
            if vector is None:
                rest.append(member)
                continue
            vectors[member.node.id] = vector
            clusterable.append(member)
        if not clusterable:
            return [], rest

        leaders: list[tuple[list[float], list[_Member]]] = []
        for member in clusterable:
            vector = vectors[member.node.id]
            best_index = -1
            best_similarity = 0.0
            for index, (leader_vector, _bucket) in enumerate(leaders):
                if len(leader_vector) != len(vector):
                    continue
                similarity = cosine_similarity(leader_vector, vector)
                # Strictly better, so an exact tie keeps the earlier-seeded
                # cluster and the walk stays a function of rank order alone.
                if similarity >= self.cluster_cosine and similarity > best_similarity:
                    best_index = index
                    best_similarity = similarity
            if best_index < 0:
                leaders.append((vector, [member]))
            else:
                leaders[best_index][1].append(member)

        groups: list[_Group] = []
        for _leader_vector, bucket in leaders:
            label = self._ctfidf_label(bucket)
            groups.append(
                _Group(
                    stage=STAGE_EMBEDDING,
                    signature=(
                        STAGE_EMBEDDING,
                        tuple(sorted(member.node.id for member in bucket)),
                    ),
                    label=label,
                    ask_hint=label,
                    members=bucket,
                    vectors={member.node.id: vectors[member.node.id] for member in bucket},
                )
            )
        return groups, rest

    def _node_vector(self, node: Node) -> list[float] | None:
        """One unit vector per node: the mean of its chunk vectors.

        Chunks first, because they are what the store keeps for every node that
        has been embedded at all, and the node-level vector second, for a
        fixture or a freshly written node whose chunks have not landed yet.
        Chunks of a width other than the first one's are dropped rather than
        averaged — vectors of two widths are points in two different spaces.
        """

        rows = [
            list(chunk.embedding)
            for chunk in self.store.list_node_chunks(node.id)
            if chunk.embedding
        ]
        if not rows and node.embedding:
            rows = [list(node.embedding)]
        if not rows:
            return None
        width = len(rows[0])
        kept = [row for row in rows if len(row) == width]
        mean = [sum(row[index] for row in kept) / len(kept) for index in range(width)]
        return _normalize_vector(mean)

    def _ctfidf_label(self, bucket: list[_Member]) -> str:
        terms = self._ranked_terms(bucket)
        if not terms:
            return _phrase_from_content(bucket[0].node.content)
        return " ".join(terms[:CTFIDF_LABEL_TERMS])

    def _ranked_terms(self, bucket: list[_Member]) -> list[str]:
        """This cluster's terms, most distinguishing first.

        ``tf * log(N / (1 + df))``: c-TF-IDF, with the cluster standing in for
        the document. Document frequencies come from the FTS vocab index, so
        rarity is measured against exactly the corpus recall searches.

        The unsmoothed numerator is what makes the label readable, and it is
        worth naming why. Under ``log(1 + N / (1 + df))`` a term present in
        *every* document still scores 0.69, which three occurrences inside the
        cluster are more than enough to turn into a winning term: a corpus of
        memory-server notes gets labelled "memory server". Plain ``log(N / (1 +
        df))`` sends a universal term negative, so it cannot win however often
        the cluster repeats it, while a term the cluster shares with three
        documents out of eleven still scores 1.0. A store predating the vocab
        index, or one with no indexed documents, degrades to plain term
        frequency rather than to nothing.
        """

        counted: Counter[str] = Counter()
        for member in bucket:
            counted.update(_terms(member.node.content))
        if not counted:
            return []

        candidates = [
            term
            for term, _count in sorted(counted.items(), key=lambda item: (-item[1], item[0]))
        ][:CTFIDF_TERM_BUDGET]

        frequencies = self._document_frequencies(candidates)
        total = self._document_count()

        scored: list[tuple[float, str]] = []
        for term in candidates:
            weight = float(counted[term])
            if total > 1 and frequencies is not None:
                weight *= log(total / (1.0 + float(frequencies.get(term, 0))))
            scored.append((weight, term))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [term for _weight, term in scored]

    def _enrich(self, head: str, bucket: list[_Member]) -> tuple[str, str]:
        """A multi-word label for a bucket that only knows one word for itself.

        Stages 1 and 2 name a cluster after a thing the corpus stores — a
        structural value, a directory segment — and the field measurement that
        motivates this is what happens when that thing is a house word: 90.8 %
        of delivered clusters in the live window were stage-2 path collapses,
        and they read as ``public (73)``, ``test (37)``, ``mmo (8)``. A row
        like that is not an invitation, it is a word.

        So the head keeps its place and the cluster's own most distinguishing
        terms are appended to it, taken from :meth:`_ranked_terms` — the
        existing c-TF-IDF ranker, reused rather than reimplemented, so a path
        label and an embedding label are distinguishing by the same measure
        against the same index. Enrichment is what *rescues* those clusters
        from the gate rather than leaving it to withhold them: ``public``
        scores 1.58 alone, and two bucket terms seen in 100 documents each add
        3.24 apiece.

        The returned ask-hint is **not** the label. It is the head plus at most
        enough terms to reach :data:`ASK_HINT_MAX_TOKENS`, so a head that is
        already two words is handed back untouched — see that constant for why
        widening it further would retune the curtail rule as a side effect.
        """

        head = _collapse(head)
        head_tokens = head.split()
        taken = set(head_tokens)
        extra: list[str] = []
        for term in self._ranked_terms(bucket):
            if term in taken:
                continue
            extra.append(term)
            taken.add(term)
            if len(extra) >= CTFIDF_LABEL_TERMS - 1:
                break
        label = _shorten(" ".join([head, *extra]), MAX_LABEL_CHARS)
        room = ASK_HINT_MAX_TOKENS - len(head_tokens)
        # A head that normalized away to nothing is dropped rather than joined
        # into a leading space; ``_cluster_of`` has the last fallback for a
        # group that ends up with no name at all.
        hint = " ".join(part for part in [head, *extra[:room]] if part) if room > 0 else head
        return label, hint

    # -- the delivery gate ---------------------------------------------

    def _reset_corpus_memo(self) -> None:
        """Forget the corpus statistics of the previous build.

        Per :meth:`build`, not per builder: the memo exists to keep one build
        from asking the same index the same question once per bucket, and a
        memo that outlived the call would answer the *next* recall out of a
        corpus that has since moved.
        """

        self._fts_total = None
        self._term_df = {}
        self._df_available = True

    def _document_count(self) -> int:
        """Indexed documents (c-TF-IDF ``N``), asked at most once per build."""

        if self._fts_total is None:
            try:
                self._fts_total = int(self.store.fts_document_count())
            except (AttributeError, sqlite3.OperationalError):  # pragma: no cover - old store
                self._fts_total = 0
        return self._fts_total

    def _document_frequencies(self, terms: Sequence[str]) -> dict[str, int] | None:
        """``term -> documents containing it``, or ``None`` if unanswerable.

        ``None`` rather than an empty mapping, because the two mean opposite
        things to a rarity measure: a term the index has never seen is
        maximally rare, while an index that cannot be read says nothing about
        rarity at all. Collapsing them would make a store with no vocabulary
        table look like a store where every label is distinguishing.
        """

        wanted = [term for term in dict.fromkeys(terms) if term not in self._term_df]
        if wanted and self._df_available:
            try:
                fetched = self.store.term_document_frequencies(wanted)
            except (AttributeError, sqlite3.OperationalError):  # pragma: no cover - old store
                self._df_available = False
            else:
                for term in wanted:
                    self._term_df[term] = int(fetched.get(term, 0))
        if not self._df_available:
            return None
        return {term: self._term_df.get(term, 0) for term in terms}

    def _label_score(self, label: str) -> float | None:
        """``S3``: the information content of ``label``'s best terms, in nats.

        ``None`` when the corpus cannot answer — see :meth:`_deliverable` for
        what happens then. Terms are taken unstemmed through :func:`_terms`,
        because the document-frequency index is ``unicode61`` and does not
        stem: a stemmed term reports ``df = 0``, scores as maximally rare, and
        turns the gate inside out.
        """

        terms = list(dict.fromkeys(_terms(label)))
        if not terms:
            return 0.0
        total = self._document_count()
        frequencies = self._document_frequencies(terms)
        if total <= 1 or frequencies is None:
            return None
        scale = float(total + 1)
        scores = sorted(
            (log(scale / (1.0 + float(frequencies.get(term, 0)))) for term in terms),
            reverse=True,
        )
        return sum(scores[:LABEL_GATE_TOP_TERMS])

    def _deliverable(self, label: str) -> bool:
        """Whether a cluster carrying ``label`` may be delivered at all.

        The whole rule, and the intent behind it restated: a cluster the
        cascade could not give a distinguishing name is withheld rather than
        shipped, because silence is cheaper than a row the agent cannot act on
        — and after the response budget has taken its share the map has only a
        handful of rows to spend, so a wasted one costs a large fraction of the
        channel.

        Withheld means *removed from delivery*, never relabelled to a catch-all
        and never appended to ``clusters`` with a marker: anything inside
        ``clusters`` is a delivered item to :func:`_delivered_items` and a
        novelty-consuming hint to the ae probe, so a "noise" row would suppress
        the curtail rule on a cluster nobody could ever follow. It goes to the
        journal (:class:`_FilterJournal`) and nowhere else.

        When the corpus cannot be measured — an old schema, an index with
        nothing in it — the gate degrades to the purely structural half of the
        predicate, ``two distinct content terms``. A store with no corpus
        statistics is a store where "generic" is undefined, and a gate that
        withholds everything on a missing index is worse than one that admits a
        weak label.
        """

        score = self._label_score(label)
        if score is None:
            return len(set(_terms(label))) >= 2
        return score >= LABEL_GATE_MIN_IC

    # -- curtail -------------------------------------------------------

    def _reoffer(
        self, verdict: _Curtailment, digest: Callable[[], str]
    ) -> tuple[bool, str | None]:
        """Whether a build under ``verdict`` offers a map, and what it persists.

        ``(True, None)`` is an open key: a map, and nothing to persist.
        ``(False, None)`` is the frozen collapse — the valve unset, or an
        armed key inside its skip run — the marker exactly as it always was.
        The other two shapes are the valve's. ``(False, digest)`` is a marker
        that carries the pool's digest: the first marker after an offer,
        which records the baseline, or a served skip run whose rebuilt pool
        matched it, which starts the run over instead of re-offering — not an
        offer, so the offer count and the run's width do not move. ``(True,
        digest)`` is a served skip run whose pool moved, or one with no
        baseline to compare against (a window the frozen code wrote): build
        the full map, and should it come back with nothing deliverable in it,
        persist the digest on the marker as the new baseline.

        ``digest`` is a thunk because computing it is the one new cost — the
        anchor lookups and the cold-lane ledger probe — and the frozen branch
        must not pay it: with the valve unset this method returns before
        calling it, so an unarmed build is byte-identical to the frozen one.
        The valve is read here and nowhere else on the collapse path.
        """

        if not verdict.collapse:
            return True, None
        if not curtail_decay_from_env():
            return False, None
        if not verdict.retry_due:
            return False, digest() if verdict.lead == 0 else None
        current = digest()
        if verdict.pool_digest is not None and current == verdict.pool_digest:
            return False, current
        return True, current

    def _collapsed(
        self,
        scope: str,
        key: str,
        *,
        pool_size: int,
        streak: int,
        selection: SelectionAccounting,
        cold: tuple[int, int] | None,
        pool_digest: str | None,
    ) -> RecallMap:
        """The collapsed form: shape kept, content dropped, streak carried."""

        return RecallMap(
            scope=scope,
            key=key,
            clusters=(),
            pool_size=pool_size,
            covered=0,
            curtailed=True,
            streak=streak,
            selection=selection,
            selection_sample_limit=_selection_sample_limit(
                (),
                pool_size,
                0,
                selection=selection,
                curtailed=True,
                streak=streak,
                pool_digest=pool_digest,
            ),
            cold=cold,
            pool_digest=pool_digest,
        )

    def _pool_digest(
        self,
        pool: Sequence[_Member],
        cold_rows: Sequence[tuple[int, Node, Any, Any]] | None,
        *,
        candidates: Sequence[tuple[int, Node, Any, dict[str, float]]] | None,
    ) -> str:
        """What this build could have said, as one comparable string.

        Three sorted id lists: the admitted members of the residual pool, the
        ``(member, anchor)`` pairs stage 3 would partition them by, and the
        cold-lane candidates the lane would rank — the eligible ones, after
        the ledger probe, because a candidate the lane would not deliver is
        not something new to say. Order-free, so two processes or two
        deliveries over the same pool agree; content-free, so the digest
        carries no text onto the wire. An unarmed lane contributes the same
        empty list an armed lane with nothing eligible does.
        """

        members = sorted(member.node.id for member in pool)
        assignment, _anchors, _rest = self._anchor_assignment(pool)
        anchors = [list(pair) for pair in sorted(assignment.items())]
        cold = (
            sorted(node.id for _ordinal, node, _result, _values in candidates)
            if cold_rows is not None and candidates
            else []
        )
        canonical = json.dumps([members, anchors, cold], separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[
            :CURTAIL_POOL_DIGEST_CHARS
        ]

    def _curtailment(
        self,
        scope: str,
        task: str | None,
        task_pattern: str | None = None,
        *,
        want_ask_follows: bool = False,
    ) -> _Curtailment:
        """This key's unaccepted-offer streak, read only when it could bite.

        The gate in front of the read is :class:`_CurtailMemo`, and the whole
        argument for it is there. What matters here: a verdict returned from
        the memo is an estimate and never collapses anything, because the
        estimate is only trusted while it stays *below* the threshold.

        ``task_pattern`` is threaded in for one reason and it is not a
        refinement: the memo must be keyed by :func:`cache_key` with *both*
        components, exactly as the map's structure cache is, or a client whose
        ``task`` changes every turn gets a fresh memo per turn while its map
        keeps hitting one cached structure — the counter and the thing it
        counts would be describing different keys.

        ``want_ask_follows`` asks the read for the ask-echo probe as well, and
        forces a read when the memo in hand was seeded without it. It buys no
        extra query — the probe is computed from rows the read already walks —
        and it changes no verdict. What it does buy is staleness of exactly
        the kind the memo already accepts: an ask that landed since the read
        is invisible until the next one, so the gate can miss a follow and
        demote a row it should have kept. That is the same window curtail
        judges by, deliberately, so the two rules cannot disagree about
        whether one delivery was followed.
        """

        key = cache_key(scope, task=task, task_pattern=task_pattern)
        memo = self._curtail_memo.get(key)
        if (
            memo is not None
            and not memo.estimate().collapse
            and (memo.ask_collected or not want_ask_follows)
        ):
            self.last_curtail_read = False
            self.last_ask_follows = memo.ask_followed
            return memo.estimate()

        self.last_curtail_read = True
        ask_follows: set[str] | None = set() if want_ask_follows else None
        verdict = self._read_curtailment(
            scope, task, task_pattern, ask_follows=ask_follows
        )
        followed = frozenset(ask_follows or ())
        self._curtail_memo.pop(key, None)
        self._curtail_memo[key] = _CurtailMemo(
            verdict=verdict,
            ask_followed=followed,
            ask_collected=want_ask_follows,
        )
        while len(self._curtail_memo) > MAX_CACHE_ENTRIES:
            self._curtail_memo.pop(next(iter(self._curtail_memo)))
        self.last_ask_follows = followed
        return verdict

    def _note_delivery(
        self,
        scope: str,
        task: str | None,
        task_pattern: str | None = None,
        *,
        offered: bool,
    ) -> None:
        """Count one map this builder just handed the server to deliver.

        ``offered`` separates a map from a collapsed marker, because only a
        map can go unaccepted. Counted here rather than inferred at read time
        so that the gate above stays a pure function of what this process
        knows it did.
        """

        memo = self._curtail_memo.get(
            cache_key(scope, task=task, task_pattern=task_pattern)
        )
        if memo is None:  # pragma: no cover - the read always seeds one
            return
        memo.total += 1
        if offered:
            memo.offers += 1

    def _read_curtailment(
        self,
        scope: str,
        task: str | None,
        task_pattern: str | None = None,
        *,
        ask_follows: set[str] | None = None,
    ) -> _Curtailment:
        """Walk this key's deliveries back until one of them was followed.

        Newest first, stopping at the first delivery with a consumed cluster:
        what is *before* that delivery cannot make the channel look unused,
        because the channel demonstrably was used. Everything walked past is
        the streak; the offers among it are what decides the collapse; the
        offerless run at the head of it is the ``lead`` the decay valve
        measures its skip quota against, and costs this walk nothing it was
        not already counting.

        Three reads for the whole window, not three per delivery: the medoid
        timestamps and the ledger's lookup verdicts each come back in one
        batch and every query is tokenized once, because this is the expensive
        half of the rule and :class:`_CurtailMemo` exists to keep it from being
        asked often. Any store that cannot answer a read — an older schema, a
        ranking-only store face — contributes no evidence from it, and a store
        that can answer none of them yields an empty verdict, which never
        collapses anything.

        ``ask_follows``, when a set is passed, is filled with the medoids whose
        label or ask-hint a later query in this window echoed — the ask-echo
        probe of :meth:`_was_followed`, reported per medoid instead of per
        delivery so :func:`_demoted` can ask it about one row. It is an out
        parameter rather than a second return value because it is a by-product:
        the verdict is what this method decides, and the harvest must not be
        able to change it. Two properties make that literal — the walk below
        covers the *whole* window rather than stopping at the streak boundary
        (a follow before the boundary is still a follow of that row), and it
        runs over the rows and tokens already in hand, so asking for it costs
        no read.
        """

        try:
            history = self._delivery_history(scope, task, task_pattern)
        except (AttributeError, sqlite3.Error):  # pragma: no cover - old store
            return _Curtailment(streak=0, offers=0)
        if not history:
            return _Curtailment(streak=0, offers=0)

        items = [_delivered_items(row.get("recall_map")) for row in history]
        try:
            accessed = self._medoid_access(items)
        except sqlite3.Error:  # pragma: no cover - defensive
            accessed = {}
        try:
            looked_up = self._medoid_lookups(history)
        except (AttributeError, sqlite3.Error):  # pragma: no cover - old store
            looked_up = {}
        queries = [token_set(str(row.get("query") or "")) for row in history]

        if ask_follows is not None:
            for index, delivered in enumerate(items):
                later = queries[:index]
                if not later:
                    continue
                for item in delivered:
                    if not item.medoid_id or item.medoid_id in ask_follows:
                        continue
                    if any(
                        _echoes(item.label, query) or _echoes(item.ask_hint, query)
                        for query in later
                    ):
                        ask_follows.add(item.medoid_id)

        streak = 0
        offers = 0
        lead = 0
        pool_digest: str | None = None
        since_digest = 0
        for index, row in enumerate(history):
            delivered = items[index]
            if delivered and self._was_followed(
                delivered,
                delivered_at=str(row.get("created_at") or ""),
                later_queries=queries[:index],
                accessed=accessed,
                looked_up=looked_up.get(str(row.get("id") or ""), frozenset()),
            ):
                break
            streak += 1
            if delivered:
                offers += 1
            elif not offers:
                lead += 1
                if pool_digest is None:
                    # The newest lead row carrying a digest is the baseline;
                    # the rows above it are the run served since. Parsed off
                    # the payload already in hand, and consulted by nothing
                    # on the frozen path.
                    payload = row.get("recall_map")
                    digest = (
                        payload.get("pd") if isinstance(payload, Mapping) else None
                    )
                    if isinstance(digest, str) and digest:
                        pool_digest = digest
                        since_digest = lead - 1
        return _Curtailment(
            streak=streak,
            offers=offers,
            lead=lead,
            pool_digest=pool_digest,
            since_digest=since_digest,
        )

    def _delivery_history(
        self, scope: str, task: str | None, task_pattern: str | None = None
    ) -> list[dict[str, Any]]:
        """This key's recent deliveries, newest first.

        The read is keyed the way :func:`cache_key` keys the map: on the
        ``task_pattern`` when there is one, on the ``task`` when there is not.
        That alignment is the whole point of this method. Before
        ``recall_events`` had a ``task_pattern`` column the read could only ask
        for the ``task``, and the docstring here called that a safe compromise
        because it merges tasks rather than splitting them — which was true of
        a stable ``task`` and false of the client that matters. An AE chat
        sends ``task=chat:<id>/turn-<N>`` per the section 9 contract, a fresh
        string every turn, so the window under a stable ``task_pattern`` key
        was not merely coarse: it was empty on every single turn, and
        :data:`CURTAIL_STREAK` could never accumulate for the one channel most
        in need of going quiet.

        Storage carries the pattern-or-legacy-task disjunction, so a row is
        counted when it names this pattern, or when it names no pattern at all
        and its ``task`` matches — the only identity a row written before the
        column can offer. The ``task`` re-check below therefore runs *only* for
        a key with no pattern, where it still does its original job: make a
        task-less key its own key instead of a bucket collecting every task in
        the scope.

        What is left is under-seeing, in two shapes, both of which end in "less
        history": a stored pattern spelled differently from this call's
        (storage matches strings, the key folds them), and a scope resolved
        differently from the one the map is built under. Seeing less history
        can only shorten a streak, and a shorter streak is a map that keeps
        being delivered, which is this module's standing answer to every
        ambiguity. A key with no pattern is affected by neither: those rows are
        matched exactly as they always were.

        One over-see survives and is named rather than hidden: a key with no
        pattern still counts rows that carry one, because the storage filter
        for that case is byte-identical to what it always was. It takes a
        client that sends a ``task_pattern`` on some calls and not others under
        the same ``task``; the rows are still that task's own deliveries, and
        the fix would be a behaviour change to every existing caller of
        :meth:`~living_memory.storage.MemoryStore.recent_recall_map_history`
        for a case no client produces.
        """

        rows = self.store.recent_recall_map_history(
            scope=scope,
            task=task,
            task_pattern=task_pattern or None,
            limit=CURTAIL_HISTORY_LIMIT,
        )
        if task_pattern:
            return [
                row for row in rows if isinstance(row.get("recall_map"), Mapping)
            ]
        wanted = normalize_key(task) if task else ""
        return [
            row
            for row in rows
            if normalize_key(str(row.get("task") or "")) == wanted
            and isinstance(row.get("recall_map"), Mapping)
        ]

    def _medoid_access(
        self, items: Sequence[Sequence[_DeliveredItem]]
    ) -> dict[str, str]:
        """``node_id -> last access timestamp`` for every medoid on offer.

        One primary-key batch for the whole window — at most
        :data:`CURTAIL_HISTORY_LIMIT` times :data:`MAX_CLUSTERS` ids, well
        inside SQLite's parameter limit — and two columns of it, because the
        live path has no business deserializing whole nodes to read one
        timestamp off each.
        """

        ids = sorted(
            {item.medoid_id for delivered in items for item in delivered if item.medoid_id}
        )
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.store.connection.execute(
            f"SELECT id, last_accessed FROM nodes WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        return {str(row["id"]): str(row["last_accessed"] or "") for row in rows}

    def _medoid_lookups(
        self, history: Sequence[Mapping[str, Any]]
    ) -> dict[str, frozenset[str]]:
        """``delivery event id -> node ids fetched inside that delivery's window``.

        Read off the frozen delivery ledger rather than recomputed here, and
        that is the point: ``lookup_consumed`` is maintained by storage over
        the same ``(delivered_at, outcome_end]`` range the M/C/K triple uses,
        so the two consumers of "was this delivery followed" cannot disagree
        about the same delivery by owning two definitions of its window.

        One primary-key probe per delivery in the window — at most
        :data:`CURTAIL_HISTORY_LIMIT` of them — because the ledger's key leads
        with the delivery event id. Measured over a full 24-delivery window of
        six-cluster maps: 0.014 ms, against 0.097 ms for the ``nodes`` batch it
        sits beside and 0.118 ms for the history read itself. It is behind the
        same :class:`_CurtailMemo` gate as both.

        Only ``lookup_consumed = 1`` is evidence. NULL is the ledger's honest
        "unobservable" for a window that closed before this database recorded
        lookups at all, and reading it as a follow would hand every historical
        delivery a free pass and switch the curtail rule off for exactly the
        corpus it was built to describe. Those deliveries keep being judged by
        the two probes that always judged them.
        """

        ids = [
            event_id
            for event_id in dict.fromkeys(
                str(row.get("id") or "") for row in history
            )
            if event_id
        ]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.store.connection.execute(
            f"""
            SELECT delivery_event_id, node_id
            FROM {RECALL_DELIVERY_HISTORY_TABLE}
            WHERE delivery_event_id IN ({placeholders})
              AND lookup_consumed = 1
            """,
            ids,
        ).fetchall()
        followed: dict[str, set[str]] = {}
        for row in rows:
            followed.setdefault(str(row["delivery_event_id"]), set()).add(
                str(row["node_id"])
            )
        return {event_id: frozenset(nodes) for event_id, nodes in followed.items()}

    @staticmethod
    def _was_followed(
        delivered: Sequence[_DeliveredItem],
        *,
        delivered_at: str,
        later_queries: Sequence[frozenset[str]],
        accessed: Mapping[str, str],
        looked_up: Collection[str] = (),
    ) -> bool:
        """Whether any cluster of one delivered map was acted on afterwards.

        Three probes, and they are not redundant. ``looked_up`` is the only
        *exogenous* one: an id-fetch of a specific ULID is an act only a reader
        who was handed that ULID performs. The other two are endogenous — a
        medoid's ``last_accessed`` is bumped by any recall that returns it, and
        the trigger boost mixes hub schemas into nearly every recall, so a hub
        reads as "used" forever on that probe alone. Both stay anyway: in
        curtail every ambiguity resolves against collapsing, and dropping a
        probe would remove evidence, not noise.

        Timestamps are compared as strings because they are stored as
        second-resolution UTC ISO-8601, where lexical order *is* chronological.
        The comparison is inclusive: at one-second resolution a tie is not
        evidence of anything, and the direction this module resolves
        non-evidence in is "used".
        """

        for item in delivered:
            if item.medoid_id and item.medoid_id in looked_up:
                return True
            touched = accessed.get(item.medoid_id, "")
            if delivered_at and touched and touched >= delivered_at:
                return True
            for query in later_queries:
                if _echoes(item.label, query) or _echoes(item.ask_hint, query):
                    return True
        return False

    # -- cache ---------------------------------------------------------

    def _corpus_revision(self) -> tuple[Any, ...]:
        """Identity of everything the cached *structure* was derived from.

        Two tiers, exactly as ``retrieval._chunk_corpus_revision``: a free
        ``(total_changes, PRAGMA data_version)`` probe answers "has anything
        been written at all", and only when it moves does the second tier ask
        the chunk and anchor tables themselves. Most recalls move the probe
        with their own bookkeeping — ``record_access`` writes on every call —
        and move neither table, so the second tier is two indexed seeks that
        return the value already held.
        """

        connection = self.store.connection
        probe = (int(connection.total_changes), _data_version(connection))
        cached = self._revision
        if (
            cached is not None
            and cached[0] == RELEVANCE_POLICY_DIGEST
            and probe == self._write_probe
        ):
            return cached
        revision = (
            RELEVANCE_POLICY_DIGEST,
            _chunk_table_revision(connection),
            tuple(self.store.query_anchor_revision()),
        )
        self._write_probe = probe
        self._revision = revision
        return revision

    def _remember(
        self, key: str, revision: tuple[Any, ...], groups: list[_Group]
    ) -> None:
        templates = tuple(
            _ClusterTemplate(
                stage=group.stage,
                signature=group.signature,
                label=group.label,
                ask_hint=group.ask_hint,
                member_ids=frozenset(member.node.id for member in group.members),
            )
            for group in groups
        )
        self._cache.pop(key, None)
        self._cache[key] = _CachedStructure(revision=revision, templates=templates)
        while len(self._cache) > MAX_CACHE_ENTRIES:
            self._cache.pop(next(iter(self._cache)))

    def _recount(
        self, templates: tuple[_ClusterTemplate, ...], pool: list[_Member]
    ) -> list[_Group] | None:
        """Refit a cached structure to the pool in hand, or refuse to.

        Stages 1 and 2 re-derive membership exactly — their signatures are pure
        functions of a node's context — so a node the cached pool never saw
        still lands in the right cluster. Stages 3 and 4 match by remembered
        membership: an anchor lookup is cheap but a *re-clustering* is not, and
        a cached embedding cluster has no rule to offer beyond the ids it was
        built from.

        That asymmetry is what :data:`CACHE_MIN_COVERAGE` guards. When the pool
        has moved far enough that the cached labels no longer describe most of
        it, this returns ``None`` and the caller rebuilds — stability is worth
        having only while it is still stability *about this pool*.
        """

        if not templates:
            return None
        ordered = sorted(
            enumerate(templates),
            key=lambda item: (_STAGE_ORDER.get(item[1].stage, 99), item[0]),
        )
        buckets: dict[int, list[_Member]] = {}
        covered = 0
        for member in pool:
            matched = False
            for index, template in ordered:
                if not self._matches(template, member.node):
                    continue
                buckets.setdefault(index, []).append(member)
                covered += 1
                matched = True
                break
            if not matched:
                # A fresh cascade may have a new structural/path/anchor
                # signature or seed a new embedding cluster. A template cache
                # has no authority to silently omit that member.
                return None
        if covered < CACHE_MIN_COVERAGE * len(pool):
            return None

        # Stage 4 is a relevance-ordered greedy walk. Re-run that membership
        # rule from fresh vectors and fresh scores, but retain cached labels
        # only if it reproduces the current template buckets exactly. This
        # keeps the cache template-only: no score, low-relevance verdict,
        # relevance order, or medoid identity survives a build boundary.
        embedding_indexes = [
            index
            for index, template in ordered
            if template.stage == STAGE_EMBEDDING and index in buckets
        ]
        embedding_vectors: dict[frozenset[str], dict[str, list[float]]] = {}
        if embedding_indexes:
            embedding_ids = {
                member.node.id
                for index in embedding_indexes
                for member in buckets[index]
            }
            embedding_members = [
                member for member in pool if member.node.id in embedding_ids
            ]
            fresh, rest = self._stage_embedding(embedding_members)
            if rest:
                return None
            current_sets = {
                frozenset(member.node.id for member in buckets[index])
                for index in embedding_indexes
            }
            fresh_sets = {
                frozenset(member.node.id for member in group.members) for group in fresh
            }
            if fresh_sets != current_sets or len(fresh) != len(embedding_indexes):
                return None
            embedding_vectors = {
                frozenset(member.node.id for member in group.members): group.vectors or {}
                for group in fresh
            }

        groups: list[_Group] = []
        for index, template in ordered:
            members = buckets.get(index)
            if not members:
                continue
            vectors = None
            if template.stage == STAGE_EMBEDDING:
                vectors = embedding_vectors.get(
                    frozenset(member.node.id for member in members)
                )
                if not vectors:
                    return None
            groups.append(
                _Group(
                    stage=template.stage,
                    signature=template.signature,
                    label=template.label,
                    ask_hint=template.ask_hint,
                    members=members,
                    vectors=vectors,
                )
            )
        return groups

    @staticmethod
    def _matches(template: _ClusterTemplate, node: Node) -> bool:
        if template.stage == STAGE_STRUCTURAL:
            key = _structural_key(node)
            return key is not None and (STAGE_STRUCTURAL, *key) == template.signature
        if template.stage == STAGE_PATH:
            # On a cold build stage 2 only sees the residual stage 1 declined.
            # Re-establish that precondition on a cache hit; otherwise a path
            # template can rename a node carrying a new structural value after
            # its directory merely because that value had no cached template.
            if _structural_key(node) is not None:
                return False
            subsystem = _node_subsystem(node)
            return subsystem is not None and (STAGE_PATH, subsystem) == template.signature
        return node.id in template.member_ids

    # -- assembly ------------------------------------------------------

    def _finish(
        self,
        groups: list[_Group],
        *,
        scope: str,
        key: str,
        pool_size: int,
        selection: SelectionAccounting,
    ) -> RecallMap | None:
        """Gate, order, cap and budget the groups into a map.

        The gate runs first and runs here rather than inside :meth:`_cluster`,
        for two reasons. It must see the label ``_disambiguate`` settled on —
        disambiguation can *lengthen* a label with a distinguishing term, which
        can only raise its score, so gating earlier would withhold clusters
        that were about to be rescued. And it must run on the cached path too:
        the structure cache stores every group it built, gate verdict included,
        because the verdict is a corpus statistic and the corpus moves under a
        cache that is deliberately validated against something else.

        What survives is ordered only by the frozen member policy: descending
        maximum member score, descending mean member score, then the unique
        best original residual ordinal. The mean uses :func:`math.fsum` over
        original residual order. Size, cascade stage, label, task and identity
        are deliberately absent from the relevance order.

        Order is deliberately not part of the cached structure.  A cached
        recount supplies this call's :class:`_Member` objects and therefore
        this call's ranks and scores; this sort and medoid tie logic refresh
        relevance-derived presentation on both cold and warm paths while
        labels and signatures remain stable.

        A pool whose every cluster is withheld yields a journal-only map.  It
        carries no delivered clusters (and therefore counts as no curtail
        offer), but it must still reach ``recall_events.recall_map``: otherwise
        the field cannot distinguish "the residual was empty" from "the label
        gate rejected everything", exactly the ambiguity the journal exists
        to remove.
        """

        populated = [group for group in groups if group.members]
        if not populated:
            return RecallMap(
                scope=scope,
                key=key,
                clusters=(),
                pool_size=pool_size,
                covered=0,
                selection=selection,
                selection_sample_limit=_selection_sample_limit(
                    (), pool_size, 0, selection=selection
                ),
            )

        # One index round trip for every label on the table, rather than one
        # per label. ``term_document_frequencies`` pays a scratch-table fold
        # per call and then seeks per token, so the batch is nearly free where
        # the calls are not — and this is the only new cost the gate puts on
        # the *cached* path, where the cascade itself no longer runs.
        self._document_frequencies(
            [term for group in populated for term in _terms(group.label)]
        )

        journal = _FilterJournal()
        deliverable: list[_Group] = []
        for group in populated:
            if self._deliverable(group.label):
                deliverable.append(group)
            else:
                journal.withheld.append(_filtered_of(FILTER_TAG_WITHHELD, group))
        journal.withheld.sort(key=lambda entry: (-entry.count, entry.label))
        if not deliverable:
            filtered = journal.block()
            return RecallMap(
                scope=scope,
                key=key,
                clusters=(),
                pool_size=pool_size,
                covered=0,
                withheld=len(journal.withheld),
                filtered=journal.names(),
                filtered_omitted=journal.omitted(),
                selection=selection,
                selection_sample_limit=_selection_sample_limit(
                    (), pool_size, 0, filtered, selection=selection
                ),
            )

        def relevance_key(group: _Group) -> tuple[float, float, int]:
            members = sorted(group.members, key=lambda member: member.rank)
            scores = [member.relevance_score for member in members]
            return (
                -max(scores),
                -(fsum(scores) / len(scores)),
                members[0].rank,
            )

        deliverable.sort(key=relevance_key)
        kept = deliverable[: self.max_clusters]
        journal.dropped.extend(
            _filtered_of(FILTER_TAG_DROPPED, group)
            for group in deliverable[self.max_clusters :]
        )
        clusters, sample_limit = self._fit(
            kept, pool_size, journal, selection=selection
        )
        if not clusters:
            # `_fit` may decide that even the breadth floor cannot carry an
            # informative gist.  It records every remaining group as dropped;
            # persist that decision instead of turning a budget refusal into
            # the same `None` an empty residual returns.
            return RecallMap(
                scope=scope,
                key=key,
                clusters=(),
                pool_size=pool_size,
                covered=0,
                dropped=len(journal.dropped),
                withheld=len(journal.withheld),
                filtered=journal.names(),
                filtered_omitted=journal.omitted(),
                selection=selection,
                selection_sample_limit=_selection_sample_limit(
                    (),
                    pool_size,
                    len(journal.dropped),
                    journal.block(),
                    selection=selection,
                ),
            )
        return RecallMap(
            scope=scope,
            key=key,
            clusters=tuple(clusters),
            pool_size=pool_size,
            covered=sum(cluster.count for cluster in clusters),
            dropped=len(journal.dropped),
            withheld=len(journal.withheld),
            filtered=journal.names(),
            filtered_omitted=journal.omitted(),
            selection=selection,
            selection_sample_limit=sample_limit,
        )

    def _fit(
        self,
        kept: list[_Group],
        pool_size: int,
        journal: _FilterJournal,
        *,
        selection: SelectionAccounting,
    ) -> tuple[list[MapCluster], int]:
        """Fit the widest common gist, trading tail breadth before gist quality.

        Every pass searches the full ``[0, MEDOID_EXAMPLE_CHARS]`` interval
        against the *current* payload.  That restart is load-bearing: the old
        shave loop could reach zero, drop several clusters, and then return the
        zero-width examples it had inherited even though the smaller map left
        roughly two hundred characters unused.

        A gist below :data:`MIN_MEDOID_EXAMPLE_CHARS` first buys room by
        dropping the tail cluster, down to :data:`MIN_CLUSTERS`.  Each such
        trade is journaled before the next search, so both the newly affordable
        example space and the journal's own cost are remeasured.  At the
        breadth floor, journal names give way one at a time; the unconditional
        withheld/dropped counts and ``names_omitted`` never do.  Only after no
        name remains may the widest affordable gist fall below the floor. The
        mandatory ``sel`` core participates in every measurement. Optional
        ``q`` samples are added only after the existing cluster shape fits, so
        they are always the first selection detail sacrificed to the budget.
        """

        active_groups = list(kept)
        # Medoid choice does not depend on the example width.  Resolve it once
        # per group — especially important for stage 4, whose true medoid is a
        # pairwise-vector calculation — and make the binary search below a
        # cheap rewrite of the quote only.
        full = [
            self._cluster_of(group, example_chars=MEDOID_EXAMPLE_CHARS)
            for group in active_groups
        ]
        breadth_floor = min(MIN_CLUSTERS, len(full))

        def at_width(example_chars: int) -> list[MapCluster]:
            return [
                replace(
                    cluster,
                    medoid=replace(
                        cluster.medoid,
                        example=_shorten(cluster.medoid.example, example_chars),
                    ),
                )
                for cluster in full
            ]

        def floor_met(clusters: Sequence[MapCluster]) -> bool:
            return all(
                bool(cluster.medoid.example)
                and len(cluster.medoid.example)
                >= min(MIN_MEDOID_EXAMPLE_CHARS, len(source.medoid.example))
                for cluster, source in zip(clusters, full, strict=True)
            )

        def widest() -> tuple[int, list[MapCluster]]:
            """Largest shared example cap whose measured payload fits."""

            low = 0
            high = MEDOID_EXAMPLE_CHARS
            best_chars = -1
            best: list[MapCluster] = []
            while low <= high:
                example_chars = (low + high) // 2
                candidates = at_width(example_chars)
                if (
                    _payload_size(
                        candidates,
                        pool_size,
                        len(journal.dropped),
                        journal.block(),
                        selection=selection,
                        selection_sample_limit=0,
                    )
                    <= MAX_RESPONSE_CHARS
                ):
                    best_chars = example_chars
                    best = candidates
                    low = example_chars + 1
                else:
                    high = example_chars - 1
            return best_chars, best

        while full:
            example_chars, clusters = widest()
            if example_chars >= 0 and floor_met(clusters):
                return clusters, _selection_sample_limit(
                    clusters,
                    pool_size,
                    len(journal.dropped),
                    journal.block(),
                    selection=selection,
                )

            if len(full) > breadth_floor:
                journal.dropped.insert(
                    0, _filtered_of(FILTER_TAG_DROPPED, active_groups.pop())
                )
                full.pop()
                continue

            if journal.drop_name():
                continue

            # Long pre-existing anchor/structural hints can make even two
            # clusters cost more than the ordinary worst-case model.  A gist
            # below the floor is not an invitation, so fail closed: move the
            # surviving breadth into the dropped journal and let `_finish`
            # persist a journal-only map. With no delivered rows competing for
            # the budget, restore the normal forensic name allowance.
            journal.dropped[0:0] = [
                _filtered_of(FILTER_TAG_DROPPED, group) for group in active_groups
            ]
            journal.name_limit = FILTER_JOURNAL_NAMES
            return [], 0
        return [], 0

    def _cluster_of(self, group: _Group, *, example_chars: int) -> MapCluster:
        medoid = self._medoid(group)
        label = _shorten(group.label, MAX_LABEL_CHARS) or _shorten(
            _phrase_from_content(medoid.node.content), MAX_LABEL_CHARS
        )
        ask_hint = _shorten(group.ask_hint, MAX_ASK_HINT_CHARS) or label
        count = len(group.members)
        return MapCluster(
            label=label,
            count=count,
            medoid=MapMedoid(
                node_id=medoid.node.id,
                example=_shorten(medoid.node.content, example_chars),
            ),
            ask_hint=ask_hint,
            plan_item=f"on touching {label} - recall '{ask_hint}' ({count})",
            stage=group.stage,
            member_ids=tuple(member.node.id for member in group.members),
        )

    @staticmethod
    def _medoid(group: _Group) -> _Member:
        """The member that speaks for the cluster.

        With vectors in play — stage 4 only, the one stage that has them — use
        the member of maximum mean similarity to the rest, which is the medoid
        proper. Everywhere else use the best original residual ordinal. That
        ordinal is unique, so it closes every tie without an identity fallback.
        Cached embedding groups carry freshly read vectors and therefore make
        the same choice as a cold group.
        """

        vectors = group.vectors
        if not vectors or len(group.members) < 2:
            return min(group.members, key=lambda member: member.rank)
        scored: list[tuple[float, int, _Member]] = []
        for member in group.members:
            own = vectors.get(member.node.id)
            if own is None:  # pragma: no cover - members always carry a vector
                continue
            others = [
                cosine_similarity(own, vectors[other.node.id])
                for other in group.members
                if other.node.id != member.node.id and other.node.id in vectors
            ]
            mean = sum(others) / len(others) if others else 0.0
            scored.append((-mean, member.rank, member))
        if not scored:  # pragma: no cover - defensive
            return min(group.members, key=lambda member: member.rank)
        return min(scored, key=lambda item: item[:2])[2]


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def cache_key(scope: str, *, task: str | None = None, task_pattern: str | None = None) -> str:
    """``(scope, normalized task or task_pattern)`` as one stable string.

    ``task_pattern`` wins when both are present: it names the recurring *class*
    of work, which is what should share a structure across sessions, while a
    ``task`` string is usually this session's phrasing of it. Both are folded by
    :func:`normalize_key`, so "postsession-extraction" and "Postsession
    Extraction" are one key.
    """

    label = task_pattern if task_pattern else task
    return f"{normalize_scope(scope)}|{normalize_key(label) if label else ''}"


def _dominant_scope(pool: list[_Member]) -> str:
    counted: Counter[str] = Counter()
    for member in pool:
        scope = getattr(member.node, "scope", "") or GLOBAL_SCOPE
        counted[str(scope)] += 1
    if not counted:
        return GLOBAL_SCOPE
    return min(counted.items(), key=lambda item: (-item[1], item[0]))[0]


def _dominant_residual_scope(results: Sequence[Any]) -> str:
    """Fallback scope when a non-empty residual admits no members."""

    counted: Counter[str] = Counter()
    for result in results:
        node = getattr(result, "node", None)
        if node is None:
            continue
        scope = getattr(node, "scope", "") or GLOBAL_SCOPE
        counted[str(scope)] += 1
    if not counted:
        return GLOBAL_SCOPE
    return min(counted.items(), key=lambda item: (-item[1], item[0]))[0]


def _filtered_of(tag: str, group: _Group) -> FilteredCluster:
    """One journal entry for a group that will not be delivered."""

    return FilteredCluster(
        tag=tag,
        label=_shorten(group.label, FILTER_JOURNAL_LABEL_CHARS),
        count=len(group.members),
    )


def _filter_block(
    withheld: int,
    dropped: int,
    names: Sequence[FilteredCluster],
    omitted: int,
) -> dict[str, Any] | None:
    """The ``filtered`` payload key, or ``None`` when nothing was filtered.

    Strictly additive, and emitted only when it has something to say, so an
    unfiltered map is byte-identical to what this module shipped before the
    journal existed. Both counts are unconditional whenever the block is
    present — that is the no-silent-caps rule in its exact form: the *names*
    are convenience and may be taken by the budget, the *numbers* never are, so
    a field reader can always tell "memory had nothing to say" (``withheld``
    high) from "the channel was full" (``dropped`` high) from "the filter is
    too harsh" (both, against a large pool).

    ``dropped`` deliberately mirrors the top-level ``more``. The duplication
    costs about fourteen characters and buys a block that is self-contained
    and greppable in a persisted payload.
    """

    if withheld <= 0 and dropped <= 0:
        return None
    block: dict[str, Any] = {"withheld": withheld, "dropped": dropped}
    if names:
        block["names"] = [entry.to_list() for entry in names]
    if omitted > 0:
        block["names_omitted"] = omitted
    return block


def _payload(
    clusters: Sequence[MapCluster],
    pool_size: int,
    dropped: int,
    *,
    curtailed: bool = False,
    streak: int = 0,
    filtered: Mapping[str, Any] | None = None,
    selection: SelectionAccounting | None = None,
    selection_sample_limit: int = SELECTION_SAMPLE_LIMIT,
    cold: tuple[int, int] | None = None,
    pool_digest: str | None = None,
) -> dict[str, Any]:
    """The wire form. Curtailed maps keep the shape and drop the content.

    ``clusters`` stays present and empty rather than being omitted, so every
    consumer of a persisted map — the instructions channel, the effect gate,
    this module's own :meth:`RecallMapBuilder._curtailment` — reads one shape
    and reaches the collapse through ``curtailed`` instead of through a
    ``KeyError``.

    ``filtered`` is a *sibling* key and never an entry inside ``clusters``.
    That is the hard rule the additive shape rests on: ``_delivered_items`` is
    paranoid about the shape of a cluster but not about extra ones, so a
    journal row inside the list would be read as a delivered item by the
    curtail probe and as a novelty-consuming hint by the ae probe — corrupting
    both on a cluster nobody was ever offered.
    """

    payload: dict[str, Any] = {
        "clusters": [cluster.to_dict() for cluster in clusters],
        "pool": pool_size,
        "covered": sum(cluster.count for cluster in clusters),
    }
    if dropped > 0:
        payload["more"] = dropped
    if filtered:
        payload["filtered"] = dict(filtered)
    if selection is not None:
        payload["sel"] = selection.to_dict(sample_limit=selection_sample_limit)
        if cold is not None:
            # Additive and unconditional-when-armed: even [0, 0] is emitted,
            # because "armed and idle" and "unarmed" are different facts. The
            # positional ``x`` vector is untouched — a cold admission is not a
            # pool admission and the accounting equation ``n = e + sum(x)``
            # holds exactly as before.
            payload["sel"]["c"] = [int(cold[0]), int(cold[1])]
    if curtailed:
        payload["curtailed"] = True
        payload["streak"] = streak
    if pool_digest is not None:
        # The decay valve's baseline, on the marker it was writing anyway. A
        # sibling key like ``filtered``: never inside ``clusters``, and read
        # back only by ``_read_curtailment``, which walks the same rows.
        payload["pd"] = pool_digest
    return payload


def _selection_sample_limit(
    clusters: Sequence[MapCluster],
    pool_size: int,
    dropped: int,
    filtered: Mapping[str, Any] | None = None,
    *,
    selection: SelectionAccounting,
    curtailed: bool = False,
    streak: int = 0,
    pool_digest: str | None = None,
) -> int:
    """Most ``q`` examples that fit without changing existing semantics."""

    maximum = min(SELECTION_SAMPLE_LIMIT, len(selection.samples))
    for sample_limit in range(maximum, -1, -1):
        payload = _payload(
            clusters,
            pool_size,
            dropped,
            curtailed=curtailed,
            streak=streak,
            filtered=filtered,
            selection=selection,
            selection_sample_limit=sample_limit,
            pool_digest=pool_digest,
        )
        if len(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        ) <= MAX_RESPONSE_CHARS:
            return sample_limit
    return 0


def _payload_size(
    clusters: Sequence[MapCluster],
    pool_size: int,
    dropped: int,
    filtered: Mapping[str, Any] | None = None,
    *,
    selection: SelectionAccounting | None = None,
    selection_sample_limit: int = SELECTION_SAMPLE_LIMIT,
    cold: tuple[int, int] | None = None,
) -> int:
    return len(
        json.dumps(
            _payload(
                clusters,
                pool_size,
                dropped,
                filtered=filtered,
                selection=selection,
                selection_sample_limit=selection_sample_limit,
                cold=cold,
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


__all__ = [
    "ASK_HINT_MAX_TOKENS",
    "CACHE_MIN_COVERAGE",
    "COLD_RANKING_CALIBRATION",
    "COLD_RANKING_MEMBERS",
    "CTFIDF_LABEL_TERMS",
    "CURTAIL_DECAY_GATE_ENV",
    "CURTAIL_DECAY_SKIP_MAX",
    "CURTAIL_HISTORY_LIMIT",
    "CURTAIL_POOL_DIGEST_CHARS",
    "CURTAIL_QUERY_OVERLAP",
    "CURTAIL_STREAK",
    "DEMOTION_REDELIVERY_WEIGHT",
    "EMBEDDING_CLUSTER_COSINE",
    "FILTER_JOURNAL_LABEL_CHARS",
    "FILTER_JOURNAL_NAMES",
    "FILTER_TAG_DROPPED",
    "FILTER_TAG_WITHHELD",
    "LABEL_GATE_MIN_IC",
    "LABEL_GATE_TOP_TERMS",
    "MAX_ASK_HINT_CHARS",
    "MAX_CLUSTERS",
    "MAX_INSTRUCTIONS_CHARS",
    "MAX_LABEL_CHARS",
    "MAX_POOL_NODES",
    "MAX_RESPONSE_CHARS",
    "MEDOID_EXAMPLE_CHARS",
    "MIN_CLUSTERS",
    "MIN_MEDOID_EXAMPLE_CHARS",
    "POOL_COLD_QUOTA_GATE_ENV",
    "POOL_COLD_SLOTS_ENV",
    "POOL_COLD_SLOTS_MAX",
    "POOL_DEMOTION_GATE_ENV",
    "POOL_DEMOTION_WINDOWS_ENV",
    "POOL_GATE_REASON_CODES",
    "POOL_USEFULNESS_FLOOR_ENV",
    "POOL_USEFULNESS_GATE_ENV",
    "RELEVANCE_CALIBRATION_MAX_BREAKPOINTS",
    "RELEVANCE_CALIBRATION_MAX_INTERPOLATION_ERROR",
    "RELEVANCE_FEATURE_MEANS",
    "RELEVANCE_FEATURE_SCALES",
    "RELEVANCE_POLICY_CALIBRATION",
    "RELEVANCE_POLICY_DIGEST",
    "RELEVANCE_POLICY_ID",
    "RELEVANCE_THRESHOLD",
    "SELECTION_LEDGER_REASON_CODES",
    "SELECTION_REASON_CODES",
    "SELECTION_SAMPLE_GIST_CHARS",
    "SELECTION_SAMPLE_LIMIT",
    "STAGE_ANCHOR",
    "STAGE_COLD",
    "STAGE_EMBEDDING",
    "STAGE_PATH",
    "STAGE_STRUCTURAL",
    "STRUCTURAL_FIELDS",
    "FilteredCluster",
    "MapCluster",
    "MapMedoid",
    "RecallMap",
    "RecallMapBuilder",
    "SelectionAccounting",
    "SelectionSample",
    "cache_key",
    "curtail_decay_from_env",
    "normalize_key",
    "pool_cold_slots_from_env",
    "pool_demotion_windows_from_env",
    "pool_usefulness_floor_from_env",
    "relevance_score",
]
