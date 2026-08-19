"""The deterministic gate. It runs after the judge and it can veto the judge.

Why the gate is not the judge
-----------------------------
A model asked "is this a durable insight?" answers yes far too often, and the
way it says yes is by writing an execution journal: *"implemented the retry
loop, added tests, all 1228 pass"*. That is the documented ``done-journal-dump``
failure mode, and it is fatal rather than merely noisy -- ``memory_recall``
ranks over everything stored, so journal traces do not sit quietly, they crowd
out the rare real lesson in every future recall. The judge therefore never has
the last word: every fact it produces is re-checked here by rules that do not
negotiate, and every rejection records a machine-readable code so the run
report says *why* rather than *how many*.

The nine rules, in the order they fire
--------------------------------------
``context_leak``
    The content carries a serialized context blob. This is a *measured* live
    defect (107 nodes affected, 2026-08-17), so it is checked first and is the
    one rule that is a safety assertion rather than a quality judgement.
``empty_fact`` / ``too_short`` / ``too_long``
    Shape. A trace nobody can act on, or a whole essay.
``journal_phrasing``
    The failure mode this module exists for. Two tiers -- see
    :data:`JOURNAL_TEMPLATES` and :data:`JOURNAL_SELF_REPORT`.
``speculation``
    Unverified plans and hedges. ``memory_remember`` says verified facts only.
``multi_fact``
    One fact per trace: enumerations and topic-disjoint paragraphs are two or
    more traces wearing one coat.
``no_concrete_anchor``
    No path, flag, symbol, error class or number -- nothing a later reader can
    match on, so recall will never surface it for the right query.
``ungrounded_identifier``
    A hard identifier the mined evidence does not contain. This is the
    anti-hallucination rule: the judge saw only the evidence span, so an
    identifier that is not in it was invented.
``rederivable_from_diff``
    Re-derivable from what the session already changed, i.e. from ``git log``.
``restates_recalled_node``
    The session was already shown this. Re-storing it is a near-duplicate the
    server's byte-identity dedup will not catch.

What is deliberately absent
---------------------------
No expected-insight text, no accept-list of facts, no session-keyed or
hash-keyed table. Every vocabulary in this module is generic phrasing:
:data:`JOURNAL_TEMPLATES` and friends list ways of *saying* "I did work", and
:data:`MECHANISM_MARKERS` lists ways of *saying* "because / returns / only
when". None of them names a fact any session was supposed to produce, which is
what makes a threshold fitted on the train split meaningful when it is reported
on sessions it has never seen.

Calibration is in ``scripts/postsession_insights.py --calibrate``: two cohorts
labelled by their producer -- machine-written node result summaries versus
agent-written ``memory_remember`` contents -- with the discrimination gap
reported on both splits so the transfer is visible rather than asserted.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from living_memory.grounding import ground_results

from .judge import default_redactor
from .mining import (
    Candidate,
    hard_identifiers,
    identifiers,
    paragraphs,
    sentences,
)
from .session import SessionRecord

__all__ = [
    "REJECTION_CODES",
    "GateConfig",
    "DEFAULT_GATE",
    "GateVerdict",
    "SessionContext",
    "Gate",
    "containment_of",
    "rejection_breakdown",
]

#: Every reason this stage may refuse a fact. The runner's report is keyed on
#: these, so adding one means adding a rule, never re-labelling an old one.
REJECTION_CODES: tuple[str, ...] = (
    "context_leak",
    "empty_fact",
    "too_short",
    "too_long",
    "journal_phrasing",
    "speculation",
    "multi_fact",
    "no_concrete_anchor",
    "ungrounded_identifier",
    "rederivable_from_diff",
    "restates_recalled_node",
    "judge_refused",
    "judge_invalid",
    "judge_unavailable",
    "duplicate_of_existing_node",
    "duplicate_within_run",
)


# --------------------------------------------------------------------------
# Rule vocabularies
# --------------------------------------------------------------------------

#: Tier 1: templates emitted by machinery, not by anybody who learned
#: something. A hit is an immediate rejection with no appeal -- these strings
#: never occur in a statement about how the world behaves.
JOURNAL_TEMPLATES = re.compile(
    r"^\s*(?:OUTCOME|RESULT|SPEC|COVERAGE|ESCALATION|SUBGOALS|MITIGATIONS)\s*:"
    r"|\bRESULT:\s*(?:DIRECT_EXECUTION|DONE|DECOMPOSE|OUT_OF_SCOPE_ESCALATION)\b"
    r"|\bOUTCOME (?:pass|fail)\b"
    r"|\bnode\s+\S{1,60}\s+(?:is\s+)?(?:done|complete[d]?)\b"
    r"|\bCompleted\b[^\n]{0,120}\bin worktree\b"
    r"|\b(?:СДЕЛАНО|ВЫПОЛНЕНО|ГОТОВО|РЕАЛИЗОВАНО|ЗАВЕРШЕНО)\b"
    r"|\bpostcondition_targets\b|\bevidence_of_completion\b"
    r"|\bChecks? passed\s*:"
    r"|\btokens used\b|\bAI Credits\b"
    # a conventional-commit subject is a commit message, i.e. `git log` itself
    r"|^\s*(?:feat|fix|chore|refactor|docs|test|perf|build|ci|style|revert)"
    r"(?:\([^)\n]{1,40}\))?!?\s*:",
    re.IGNORECASE | re.MULTILINE,
)

#: Verbs whose subject is the session itself. English past tense and past
#: participle, plus the bare imperative that node summaries are written in,
#: plus their Russian equivalents -- the corpus is bilingual and the AE
#: runtime's own summaries switch language per node.
_WORK_VERBS = (
    r"re-?(?:verified|ran|recorded|built|worked|wrote)"
    r"|added|implemented|created|wrote|built|made|refactored|removed|deleted"
    r"|migrated|updated|fixed|repaired|resolved|unblocked|silenced|downgraded"
    r"|landed|shipped|committed|amended|renamed|replaced|skipped|backfilled"
    r"|introduced|extended|wired|plumbed|ported|documented|verified|validated|ran"
    r"|switched|split|merged|restored|reverted|applied|enabled|disabled|hardened"
    r"|tightened|bumped|pinned|sealed|froze|frozen|published|generated|recorded"
    r"|captured|instrumented|refreshed|rebuilt|reproduced|reviewed|closed"
    r"|finished|completed|delivered|adjusted|corrected|cleaned|dropped|moved"
    r"|exposed|surfaced|taught|scoped|rewrote|reworked|deduplicated|deduped"
    r"|normalized|consolidated|retired|archived|promoted|demoted|propagated"
    r"|annotated|formalized|parameterized|unified"
    r"|add|implement|create|build|fix|update|refactor|remove|freeze|seal|wire"
    r"|apply|verify|document|publish|extend|replace|rename|migrate|port|introduce"
    r"|teach|harden|tighten|pin|record|capture|generate|rebuild|reproduce"
    r"|добавил\w*|добавлен\w*|создал\w*|создан\w*|перенес\w*|перенёс\w*"
    r"|реализовал\w*|реализован\w*|починил\w*|починен\w*|исправил\w*|исправлен\w*"
    r"|обновил\w*|обновлён\w*|обновлен\w*|написал\w*|написан\w*|удалил\w*"
    r"|удалён\w*|заменил\w*|заменён\w*|зафиксировал\w*|зафиксирован\w*"
    r"|провёл|провел|переснял|снял|закрыл\w*|закрыт\w*|завершил\w*|завершён\w*"
    r"|собрал\w*|собран\w*|настроил\w*|настроен\w*|внедрил\w*|внедрён\w*"
    r"|поправил\w*|доработал\w*|отревьюил\w*|поддержан\w*|поддержал\w*"
    r"|актуализирован\w*|актуализировал\w*|пройден\w*|выполнен\w*|заведён\w*"
    r"|переоткрыл\w*|прогнал\w*|запустил\w*|проверил\w*|сверил\w*|вынес\w*"
    r"|вынесен\w*|разбил\w*|разбит\w*|объединил\w*|откатил\w*|восстановил\w*"
    r"|сделал\w*|перевёл\w*|переименован\w*|переработан\w*"
    # bare imperatives / infinitives -- the mood AE node summaries are set in
    r"|заведи|добавь|добавить|сделай|сделать|исправь|исправить|приведи|привести"
    r"|перенеси|перенести|реализуй|реализовать|убери|убрать|поддержи|поддержать"
    r"|почини|починить|обнови|обновить|напиши|написать|собери|собрать|проверь"
    r"|проверить|вынеси|вынести|разбей|разбить|замени|заменить|включи|включить"
    r"|отключи|отключить|заморозь|заморозить|дай|дать"
)

#: Optional decoration a node summary opens with before its verb.
_OPENING_NOISE = r"(?:[-—–*>#\s]|\*\*|`|\[[^\]\n]{1,60}\]|summary\s*:|итог\s*:)*"

#: Tier 2: the content *opens* with a work verb -- "Added the React data
#: layer...", "Перенесены TypeScript конфиги...", "Implement the baseline
#: calculator...". This is the exact shape of a merge-commit subject, and it
#: is a journal entry no matter how many identifiers follow. No appeal.
JOURNAL_OPENING = re.compile(
    rf"\A{_OPENING_NOISE}(?:(?:I|We|Я|Мы)\s+(?:have\s+|now\s+)?)?(?:{_WORK_VERBS})\b",
    re.IGNORECASE,
)

#: Tier 2b: a claim that the work is finished or that the checks are green.
#: Also no appeal -- "all 88 adversarial tests pass" says nothing a future
#: reader can act on, and ``git log`` says it better.
JOURNAL_COMPLETION = re.compile(
    r"\b(?:all(?: \d+)?|the) (?:tests?|checks?|suites?|gates?)\b[^.\n]{0,60}"
    r"\b(?:pass|passed|passing|green|succeed|succeeded)\b"
    r"|\b\d+ (?:tests?|checks?|suites?) (?:now )?(?:pass|passed|passing|green)\b"
    r"|\b(?:tests?|checks?|suite|gates?) (?:are |is )?(?:now )?green\b"
    r"|\b(?:goal|task|node|subgoal|ticket|MR|PR) (?:is )?(?:fully )?"
    r"(?:met|done|complete[d]?|satisfied|closed|delivered|approved)\b"
    r"|\b(?:original goal|postcondition) is (?:fully )?(?:met|satisfied)\b"
    r"|\bno (?:new )?(?:code )?(?:edits?|changes?) (?:were )?needed\b"
    r"|\bzero (?:errors?|failures?|regressions?)\b"
    r"|\bпроверк\w+[^.\n]{0,40}(?:прошл\w+|зелён\w+|зелен\w+|выполнен\w+)\b"
    r"|\b(?:все |всё )?(?:тесты|проверки|чеки|гейты)[^.\n]{0,40}"
    r"(?:прошли|прошла|зелён\w+|зелен\w+|пройден\w+)\b"
    r"|\bцел\w+[^.\n]{0,40}(?:выполнен\w+|закрыт\w+|достигнут\w+|реализован\w+)\b"
    r"|\bпрошл\w+ без (?:blocking |критич\w+ )?(?:failures?|ошибок|замечаний)\b"
    r"|\b\d+/\d+\b[^.\n]{0,20}(?:зелён\w+|зелен\w+|passed|passing|green)\b"
    r"|\bacceptance (?:checks?|tests?)\b[^.\n]{0,40}\b(?:pass|passed|green)\b",
    re.IGNORECASE,
)

#: Tier 3: work-report phrasing anywhere in the text. These *can* appear
#: inside a real lesson ("the fix is a hard reboot"), so a hit only rejects
#: when the text carries no mechanism -- see :data:`MECHANISM_MARKERS`.
JOURNAL_SELF_REPORT = re.compile(
    r"\b(?:implemented|refactored|re-?ran the (?:suite|tests?))\b"
    r"|\badded (?:a |an |the )?(?:new )?(?:tests?|test case|coverage|function|method"
    r"|class|module|file|flag|option|field|parameter|helper|fixture)\b"
    r"|\b(?:wrote|added) \d+ tests?\b"
    r"|\b(?:landed|shipped|merged) (?:in|on|to)\b"
    r"|\bcommitted\b\s*`?[0-9a-f]{7,40}"
    r"|\bдобавил\w* (?:тест|покрытие|функци|метод|класс|модуль|файл|флаг|поле)",
    re.IGNORECASE | re.MULTILINE,
)

#: A statement carries a *mechanism* when it says how or why the world behaves
#: as it does, rather than what the session did. This is what buys a tier-2 hit
#: its appeal: "fixed X" is a journal entry, "X fails when Y because Z" is not.
MECHANISM_MARKERS = re.compile(
    r"\b(?:because|since|so that|which means|the reason|root cause|caused by"
    r"|due to|only if|only when|unless|otherwise|fails when|breaks when"
    r"|returns|raises|throws|requires|expects|silently|must not|never|always"
    r"|defaults? to|falls back|overrides?|takes precedence|is ignored"
    r"|потому что|поэтому|причина|из-за|иначе|возвращает|требует|игнорирует"
    r"|молча|по умолчанию|перекрывает|падает(?: с| при)|ломается)\b"
    r"|->|→|=>",
    re.IGNORECASE,
)

#: Hedges and plans. ``memory_remember``'s own policy is "verified facts only,
#: never speculation, never unverified plans".
SPECULATION_MARKERS = re.compile(
    r"\b(?:probably|likely|might(?: be)?|may be|maybe|perhaps|I think|I believe"
    r"|I suspect|seems to|appears to|presumably|could be|should probably"
    r"|we (?:will|should|plan to|intend to)|next step|TODO|to be (?:done|verified"
    r"|confirmed|checked)|not yet (?:verified|confirmed|tested)|unverified"
    r"|вероятно|возможно|наверн\w+|скорее всего|предположительно|кажется"
    r"|надо будет|планиру\w+|следующий шаг|не проверено|пока не ясно)\b",
    re.IGNORECASE,
)

#: Evidence that the hedge was resolved. A sentence may say "we suspected X,
#: measured it, and it is false" -- that is a refutation, not speculation.
VERIFICATION_MARKERS = re.compile(
    r"\b(?:measured|verified|confirmed|observed|reproduced|proved|disproved"
    r"|refuted|turned out|in fact|actually"
    r"|замер\w*|проверен\w*|подтвержден\w*|воспроизвед\w*|оказалось|опроверг\w*)\b",
    re.IGNORECASE,
)

#: Context keys the live server stamps. Seeing them inside ``content`` means a
#: malformed call serialized the whole context block into the trace body.
CONTEXT_LEAK_KEYS: tuple[str, ...] = (
    "transport_session_id",
    "lesson_kind",
    "ambient_context",
    "recall_event_id",
    "postcondition_targets",
)

#: Keys that also occur in perfectly legitimate quoted JSON, so one of them is
#: not evidence of anything. Three of them inside one object literal is.
_AMBIGUOUS_CONTEXT_KEYS: tuple[str, ...] = (
    "scope",
    "agent",
    "task",
    "session_id",
    "timestamp",
    "trigger",
    "topic",
    "files",
)

_CONTEXT_LEAK = re.compile(
    r"[\"'](" + "|".join(re.escape(key) for key in CONTEXT_LEAK_KEYS) + r")[\"']\s*:\s*[\"'{\[]"
)

_JSON_OBJECT = re.compile(r"\{[^{}]{20,2000}\}", re.DOTALL)
_QUOTED_KEY = re.compile(r"[\"'](\w+)[\"']\s*:")

_LIST_ITEM = re.compile(r"^\s*(?:[-*•‣]|\d+[.)])\s+\S", re.MULTILINE)
_INLINE_ENUM = re.compile(r"\(\s*[1-9]\s*\)|\b[1-9]\)\s")


# --------------------------------------------------------------------------
# Configuration, verdicts, session context
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Thresholds. Tuned on the ``train`` split, reported on ``eval``.

    Each default below was picked against the two control cohorts described in
    ``scripts/postsession_insights.py --calibrate``: machine-generated node
    result summaries (which the gate must reject) and agent-written
    ``memory_remember`` contents (which it must mostly keep).
    """

    min_chars: int = 60
    max_chars: int = 1400
    #: Paragraphs whose identifier sets barely overlap are separate facts.
    max_paragraphs: int = 2
    paragraph_overlap_min: float = 0.15
    #: Enumerations of this many items are a list of facts, not one fact.
    max_list_items: int = 2
    min_hard_identifiers: int = 1
    #: Hard identifiers absent from the mined evidence. Zero: the judge saw
    #: only the evidence, so anything else came from nowhere.
    max_ungrounded_identifiers: int = 0
    #: IDF containment in the session's own diff above which the fact is just
    #: the diff restated.
    diff_containment_max: float = 0.60
    #: IDF containment in any single node the session already recalled.
    recalled_containment_max: float = 0.70
    #: Journal tier 2 needs a mechanism to survive; set False to disable the
    #: appeal entirely (stricter, used by the calibration sweep).
    allow_mechanism_appeal: bool = True


DEFAULT_GATE = GateConfig()


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """Accepted, or refused with a code from :data:`REJECTION_CODES`."""

    accepted: bool
    code: str = ""
    detail: str = ""
    signals: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "code": self.code,
            "detail": self.detail,
            "signals": dict(self.signals),
        }


@dataclass(frozen=True, slots=True)
class SessionContext:
    """What the gate needs to know about the session behind a fact.

    Built once per session: tokenizing a 200 kB diff per candidate would
    dominate the run, and both corpora are identical for every candidate of
    the same session.
    """

    diff_text: str = ""
    recalled: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record: SessionRecord, *, max_diff_chars: int = 400_000) -> "SessionContext":
        """Collect the session's own diff and the nodes it was actually shown.

        Only *added* diff lines count as "already in the diff": a fact about
        code the session deleted is still worth keeping, and context lines
        belong to neither side.
        """

        chunks: list[str] = []
        size = 0
        for mutation in record.file_mutations:
            piece = mutation.path
            body = mutation.unified_diff or mutation.post_image or ""
            if mutation.unified_diff:
                body = "\n".join(
                    line[1:]
                    for line in body.splitlines()
                    if line.startswith("+") and not line.startswith("+++")
                )
            piece = f"{piece}\n{body}"
            if size + len(piece) > max_diff_chars:
                break
            chunks.append(piece)
            size += len(piece)
        recalled = {
            node.node_id: node.content
            for recall in record.recalls
            for node in recall.delivered
            if node.content
        }
        return cls(diff_text="\n".join(chunks), recalled=recalled)


def containment_of(fact: str, corpus: str) -> float:
    """IDF-weighted share of ``fact``'s tokens that ``corpus`` already carries.

    Reuses :mod:`living_memory.grounding` rather than re-implementing the
    arithmetic, so "this fact is contained in that text" means exactly what the
    live credit-assignment loop means by it. The fact plays the *result* role
    and the corpus plays the *trace* role, which is the direction that answers
    "is this fact already there".
    """

    if not fact.strip() or not corpus.strip():
        return 0.0
    graded = ground_results(corpus, {"fact": fact}, min_containment=1.1)
    verdict = graded.get("fact")
    return round(verdict.containment, 6) if verdict else 0.0


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


class Gate:
    """Applies :data:`REJECTION_CODES` in a fixed order; first hit wins.

    Fixed order matters for the report: a fact that is both a journal entry and
    re-derivable from the diff must always be counted under the same code, or
    the per-reason breakdown drifts with unrelated changes.
    """

    def __init__(
        self,
        config: GateConfig = DEFAULT_GATE,
        *,
        redactor: Callable[[str], str] = default_redactor,
    ) -> None:
        self.config = config
        # The judge sees the *redacted* payload, so a fact may legitimately
        # say `~/p/lm/x.py` where the raw transcript says
        # `/home/<user>/p/lm/x.py`. Grounding compares against both spellings
        # or the anti-hallucination rule would fire on every absolute path.
        self.redactor = redactor

    # -- individual rules -------------------------------------------------

    def _context_leak(self, fact: str) -> GateVerdict | None:
        """The measured write defect: a whole context JSON inside ``content``.

        Checked structurally rather than by pattern alone -- a fact that parses
        as a JSON object with context keys is the exact shape of the 107
        damaged nodes, and no legitimate trace is a bare JSON object.
        """

        stripped = fact.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                return GateVerdict(
                    False, "context_leak", "content is a serialized JSON object"
                )
        hit = _CONTEXT_LEAK.search(fact)
        if hit is not None:
            return GateVerdict(
                False,
                "context_leak",
                f"content embeds a serialized context key: {hit.group(1)}",
            )
        # A trace may legitimately quote one JSON key. The damaged nodes carry
        # the *whole* context dict, so require several of the ambiguous keys
        # inside one object literal before calling it a leak.
        for match in _JSON_OBJECT.finditer(fact):
            keys = set(_QUOTED_KEY.findall(match.group(0)))
            shared = keys & set(_AMBIGUOUS_CONTEXT_KEYS)
            if len(shared) >= 3:
                return GateVerdict(
                    False,
                    "context_leak",
                    "content embeds a serialized context object: "
                    + ", ".join(sorted(shared)[:5]),
                    {"context_keys": sorted(shared)},
                )
        return None

    def _shape(self, fact: str) -> GateVerdict | None:
        stripped = fact.strip()
        if not stripped:
            return GateVerdict(False, "empty_fact", "judge returned no fact")
        if len(stripped) < self.config.min_chars:
            return GateVerdict(
                False, "too_short", f"{len(stripped)} < {self.config.min_chars} chars"
            )
        if len(stripped) > self.config.max_chars:
            return GateVerdict(
                False, "too_long", f"{len(stripped)} > {self.config.max_chars} chars"
            )
        return None

    def _journal(self, fact: str) -> GateVerdict | None:
        template = JOURNAL_TEMPLATES.search(fact)
        if template is not None:
            return GateVerdict(
                False,
                "journal_phrasing",
                f"execution-journal template: {template.group(0).strip()[:60]!r}",
                {"tier": "template"},
            )
        opening = JOURNAL_OPENING.search(fact)
        if opening is not None:
            return GateVerdict(
                False,
                "journal_phrasing",
                f"opens with a work verb: {opening.group(0).strip()[:60]!r}",
                {"tier": "opening"},
            )
        completion = JOURNAL_COMPLETION.search(fact)
        if completion is not None and not (
            self.config.allow_mechanism_appeal and MECHANISM_MARKERS.search(fact)
        ):
            # The appeal is what separates "all tests pass" (a completion
            # claim) from "44 tests pass in 12.4s because jest resolves
            # related tests transitively" (a measurement with its mechanism).
            return GateVerdict(
                False,
                "journal_phrasing",
                f"completion claim: {completion.group(0).strip()[:60]!r}",
                {"tier": "completion"},
            )
        self_report = JOURNAL_SELF_REPORT.search(fact)
        if self_report is None:
            return None
        if self.config.allow_mechanism_appeal and MECHANISM_MARKERS.search(fact):
            return None
        return GateVerdict(
            False,
            "journal_phrasing",
            f"self-reported work with no mechanism: {self_report.group(0).strip()[:60]!r}",
            {"tier": "self_report"},
        )

    def _speculation(self, fact: str) -> GateVerdict | None:
        hit = SPECULATION_MARKERS.search(fact)
        if hit is None:
            return None
        if VERIFICATION_MARKERS.search(fact):
            return None
        return GateVerdict(
            False,
            "speculation",
            f"unverified hedge: {hit.group(0).strip()[:40]!r}",
        )

    def _multi_fact(self, fact: str) -> GateVerdict | None:
        """One fact per trace.

        Two independent detectors, because two shapes of "several facts" occur:
        an explicit enumeration, and paragraphs that talk about disjoint sets
        of identifiers. Prose length alone is *not* a signal -- a single fact
        with its mechanism and its why is legitimately several sentences.
        """

        items = _LIST_ITEM.findall(fact)
        if len(items) > self.config.max_list_items:
            return GateVerdict(
                False,
                "multi_fact",
                f"{len(items)} enumerated items",
                {"list_items": len(items)},
            )
        if len(_INLINE_ENUM.findall(fact)) > self.config.max_list_items:
            return GateVerdict(False, "multi_fact", "inline enumeration")

        blocks = paragraphs(fact)
        if len(blocks) > self.config.max_paragraphs:
            return GateVerdict(
                False,
                "multi_fact",
                f"{len(blocks)} paragraphs",
                {"paragraphs": len(blocks)},
            )
        if len(blocks) >= 2:
            sets = [set(hard_identifiers(block)) for block in blocks]
            first = sets[0]
            for index, other in enumerate(sets[1:], start=1):
                if not first or not other:
                    continue
                overlap = len(first & other) / len(first | other)
                if overlap < self.config.paragraph_overlap_min:
                    return GateVerdict(
                        False,
                        "multi_fact",
                        f"paragraph {index} shares {overlap:.2f} of its identifiers "
                        "with the first",
                        {"paragraph_overlap": round(overlap, 3)},
                    )
        return None

    def _anchors(self, fact: str) -> tuple[GateVerdict | None, tuple[str, ...]]:
        found = hard_identifiers(fact)
        if len(found) < self.config.min_hard_identifiers:
            return (
                GateVerdict(
                    False,
                    "no_concrete_anchor",
                    "no path, flag, symbol, error class or number",
                ),
                found,
            )
        return None, found

    def _grounded(self, found: Sequence[str], candidate: Candidate) -> GateVerdict | None:
        """Every hard identifier must occur in the mined evidence, verbatim.

        The judge is shown the evidence span and nothing else, so an identifier
        that is not in the span was invented. A substring test is deliberate:
        the evidence is raw transcript text, so ``storage.py`` inside
        ``src/living_memory/storage.py`` counts as present.
        """

        raw = f"{candidate.evidence_text}\n{' '.join(candidate.identifiers)}"
        corpus = f"{raw}\n{self.redactor(raw)}"
        missing = [token for token in found if token not in corpus]
        if len(missing) > self.config.max_ungrounded_identifiers:
            return GateVerdict(
                False,
                "ungrounded_identifier",
                f"not in the evidence span: {', '.join(sorted(missing)[:5])}",
                {"ungrounded": sorted(missing)[:10]},
            )
        return None

    def _rederivable(self, fact: str, context: SessionContext) -> GateVerdict | None:
        if not context.diff_text:
            return None
        value = containment_of(fact, context.diff_text)
        if value >= self.config.diff_containment_max:
            return GateVerdict(
                False,
                "rederivable_from_diff",
                f"containment {value:.2f} >= {self.config.diff_containment_max:.2f}",
                {"diff_containment": value},
            )
        return None

    def _restates_recalled(self, fact: str, context: SessionContext) -> GateVerdict | None:
        worst = 0.0
        worst_id = ""
        for node_id, content in context.recalled.items():
            value = containment_of(fact, content)
            if value > worst:
                worst, worst_id = value, node_id
        if worst >= self.config.recalled_containment_max:
            return GateVerdict(
                False,
                "restates_recalled_node",
                f"restates node {worst_id} (containment {worst:.2f})",
                {"recalled_containment": worst, "node_id": worst_id},
            )
        return None

    # -- entry point ------------------------------------------------------

    def check(
        self,
        fact: str,
        *,
        candidate: Candidate,
        context: SessionContext,
    ) -> GateVerdict:
        """Verdict on one judge-produced fact. Never raises."""

        for verdict in (
            self._context_leak(fact),
            self._shape(fact),
            self._journal(fact),
            self._speculation(fact),
            self._multi_fact(fact),
        ):
            if verdict is not None:
                return verdict

        anchor_verdict, found = self._anchors(fact)
        if anchor_verdict is not None:
            return anchor_verdict
        for verdict in (
            self._grounded(found, candidate),
            self._rederivable(fact, context),
            self._restates_recalled(fact, context),
        ):
            if verdict is not None:
                return verdict

        return GateVerdict(True, "", "", {"identifiers": list(found[:10])})

    def check_text(self, fact: str) -> GateVerdict:
        """Text-only rules, for calibrating against a corpus with no session.

        Used by ``--calibrate``: the journal / speculation / multi-fact /
        anchor rules are properties of the text alone, and they are the ones
        whose thresholds need a held-out cohort to justify them.
        """

        for verdict in (
            self._context_leak(fact),
            self._shape(fact),
            self._journal(fact),
            self._speculation(fact),
            self._multi_fact(fact),
        ):
            if verdict is not None:
                return verdict
        anchor_verdict, found = self._anchors(fact)
        if anchor_verdict is not None:
            return anchor_verdict
        return GateVerdict(True, "", "", {"identifiers": list(found[:10])})


def rejection_breakdown(verdicts: Iterable[GateVerdict]) -> dict[str, int]:
    """Counts per code, with every known code present so a zero is visible."""

    counts = {code: 0 for code in REJECTION_CODES}
    counts["accepted"] = 0
    for verdict in verdicts:
        if verdict.accepted:
            counts["accepted"] += 1
        else:
            counts[verdict.code] = counts.get(verdict.code, 0) + 1
    return counts


def identifier_profile(text: str) -> dict[str, list[str]]:
    """Grouped identifiers -- exposed for report readability, not for rules."""

    return {name: list(tokens) for name, tokens in identifiers(text).items()}


def sentence_count(text: str) -> int:
    """Substantive sentences. Reported, never a rule on its own."""

    return sum(1 for item in sentences(text) if len(item.split()) >= 5)
