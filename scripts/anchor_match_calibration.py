#!/usr/bin/env python3
"""Calibrate the query-anchor *match* entry on a temporally disjoint set.

``ANCHOR_MATCH_COSINE_THRESHOLD`` shipped at 0.80 as a documented default,
picked to sit under the 0.95 dedup constant rather than measured against match
recall. The scored holdout then found it admits almost nothing: 13 of 343
held-out queries, 0 of 38 ``cross_lingual``, 1 of 36 ``role_query``
(``result.md`` §5). This script measures where the floor should sit --
**without ever reading the scored goldset**.

The nested temporal split
-------------------------
The scored run is ``anchors < 2026-07-15`` scored on ``queries >= 2026-07-15``.
This calibration nests one split *inside* the scored run's anchor window::

    |-------- calibration anchors --------|-- calibration queries --|-- SCORED --|
    <                       2026-06-10                     2026-07-15           >

so every calibration query is an event that helped *build* the scored run's
anchor corpus, and is therefore disjoint from the scored, post-cutoff items by
construction. Anchors come from ``backfill_query_anchors.py backfill --until
2026-06-10T00:00:00Z``; the shape of the problem is preserved (a query facing an
anchor corpus that predates it) one window earlier.

That window yields only the ``content_grounded`` class, which is not the class
in question. The two classes that are -- ``cross_lingual`` and ``role_query`` --
carry no timestamp at all (``source_event_id: null``), so a temporal split
cannot separate them from the scored items. They are therefore **newly
authored** here, from pre-2026-07-15 traffic, under the same recorded rules the
scored items were built with (``artifacts/harness/seed-queries.json``:
``cross_lingual_rule``, ``role_query_rule``), and their disjointness from the
scored items is **asserted, not assumed** -- see ``purity`` below.

Purity
------
``--scored-goldset`` files are opened for exactly one purpose: to *exclude*.
:func:`load_exclusions` reads them and returns opaque keys -- recall
fingerprints, source event ids, normalised query hashes, and an embedding matrix
of the scored queries. It never returns, prints, or stores a scored query's
text, its relevant node ids, or any score. A calibration item that collides on
any key, or that sits within ``--near-dup-cosine`` of any scored query, is
dropped. Exclusion can only ever *remove* information; it cannot leak an answer
key into the choice of threshold, which selection on the scored set would.

What is published per swept value
---------------------------------
``coverage``   share of calibration queries with at least one anchor at or
               above the floor;
``precision``  share of *matched anchors* holding an edge to a node the item's
               relevance rule marks relevant (for ``content_grounded`` that rule
               is the grounding verdict; for the two curated strata it is the
               stratum's own resolution rule, which is stated rather than
               conflated);
``quality``    hit@1 / hit@5 / hit@10 / MRR from the real
               ``MemoryRecallService`` on the calibration snapshot;
``cost``       matched share, anchor seeds per matched query, measured recall
               wall time, and the p50 paired delta this match rate projects
               against the +5 ms budget using the per-stratum medians already
               measured in ``artifacts/anchors/latency.json``.

Every database this script opens is a snapshot or a copy of one. It never
touches ``~/.local/share/living-memory/global.sqlite3``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:  # pragma: no cover - CLI convenience
    sys.path.insert(0, str(REPO_ROOT / "src"))

from living_memory import retrieval as retrieval_module
from living_memory.config import MemoryConfig
from living_memory.embeddings import cosine_similarity
from living_memory.query_anchors import match_anchors
from living_memory.retrieval import GRAPH_SEED_LIMIT, MemoryRecallService
from living_memory.retrieval_harness import (
    GoldsetBuildConfig,
    GoldsetItem,
    ModelVisibleSpan,
    backup_database,
    build_content_grounded,
    build_cross_lingual,
    build_role_query,
    dump_goldset,
    load_goldset,
    normalize_cutoff,
    sha256_file,
)
from living_memory.storage import MemoryStore, recall_fingerprint

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Anchors are built strictly before this instant; calibration queries start
#: here. One window earlier than the scored run's 2026-07-15 cutoff.
DEFAULT_WINDOW_START = "2026-06-10T00:00:00Z"

#: Calibration queries stop here -- the scored run's cutoff. Everything at or
#: after it belongs to the scored holdout and is out of bounds.
DEFAULT_WINDOW_END = "2026-07-15T00:00:00Z"

#: Sampling seed for the ``content_grounded`` stratum. Fixed so the set is
#: reproducible; it selects events, never thresholds.
DEFAULT_SEED = 20260818

#: Cap on ``content_grounded`` items, so the class the anchors were built from
#: does not drown the two classes the calibration is actually about.
DEFAULT_CONTENT_GROUNDED_CAP = 180

#: A calibration query within this cosine of any scored query is dropped as a
#: near-duplicate. Deliberately at the dedup constant: two queries that close
#: are the same question, and one of them is in the answer key.
DEFAULT_NEAR_DUP_COSINE = 0.95

DEFAULT_THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
DEFAULT_LIMITS = (3, 5, 10)

#: Ranked anchors pulled per query in the single measurement pass. Well above
#: the largest swept limit, so every (threshold, limit) cell is derivable from
#: one pass instead of one scan each.
RANK_DEPTH = 30

#: Per-stratum paired p50 deltas measured in ``artifacts/anchors/latency.json``
#: on the live corpus. Used only to *project* a cost for a match rate; the
#: measured wall time of each sweep run is published beside the projection.
LATENCY_MATCHED_P50_MS = 8.255
LATENCY_UNMATCHED_P50_MS = -0.637
LATENCY_BUDGET_MS = 5.0

_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Authored calibration specs
#
# Both lists are authored HERE rather than borrowed: ``seed-queries.json`` holds
# exactly the 38 + 36 items already inside the scored goldset, so there is no
# unused surplus to draw on. Each obeys its recorded rule, and every item is
# re-verified mechanically at build time (see ``_filter_cross_lingual``).
# ---------------------------------------------------------------------------

#: ``cross_lingual_rule``: a Russian jargon query against its plain-English
#: paraphrase, relevance resolved MECHANICALLY as the paraphrase's top-1 on the
#: snapshot. Kept only when the paraphrase's top-1 vector score >= 0.40, the
#: node's content is predominantly Latin script, the jargon query scores
#: strictly lower on that node and does not already rank it first, and the node
#: is topically about the query's subject. 151 pairs were authored and measured
#: against the calibration snapshot; the ones below are what survived.
CROSS_LINGUAL_SPECS: tuple[dict[str, Any], ...] = (
    {"jargon_query": "затухание узлов памяти как устроено",
     "paraphrase_query": "how does memory node decay work", "scope": "project:lm"},
    {"jargon_query": "консолидация трейсов в схемы",
     "paraphrase_query": "how are schemas consolidated from traces", "scope": "project:lm"},
    {"jargon_query": "скоуп памяти как резолвится от узкого к широкому",
     "paraphrase_query": "how is the memory scope resolved from narrowest to broadest",
     "scope": "project:lm"},
    {"jargon_query": "супёрсид узла при правке факта",
     "paraphrase_query": "how memory_teach supersedes a stale fact", "scope": "project:lm"},
    {"jargon_query": "адаптивные веса каналов как учатся",
     "paraphrase_query": "how adaptive retrieval channel weights are learned online",
     "scope": "project:lm"},
    {"jargon_query": "заземление потребления и фидбек",
     "paraphrase_query": "how grounded recall feedback is applied to nodes", "scope": "project:lm"},
    {"jargon_query": "разбиение длинного узла на куски для вектора",
     "paraphrase_query": "how a long node is split into chunks for embedding", "scope": "project:lm"},
    {"jargon_query": "временная подсказка когда узел нужен",
     "paraphrase_query": "the temporal hint that says when a node is needed", "scope": "project:lm"},
    {"jargon_query": "почему прогон на живой базе опасен без копии",
     "paraphrase_query": "why running an experiment against the live database without a snapshot is unsafe",
     "scope": "project:lm"},
    {"jargon_query": "эластик индексы логов k8s продакшн",
     "paraphrase_query": "which elasticsearch indices hold kubernetes production logs",
     "scope": "project:online"},
    {"jargon_query": "сёрчбар дашборда липнет при скролле",
     "paraphrase_query": "the dashboard searchbar sticky scroll behaviour", "scope": "project:online"},
    {"jargon_query": "эксперимент с тултипом на карте",
     "paraphrase_query": "the map tooltip experiment", "scope": "project:online"},
    {"jargon_query": "шапка дашборда и погода города",
     "paraphrase_query": "the dashboard header and the city weather block", "scope": "project:online"},
    {"jargon_query": "почему галочка остаётся выключенной при сохранении",
     "paraphrase_query": "why the checkbox stays disabled when saving", "scope": "project:online"},
    {"jargon_query": "события аналитики на показ и клик",
     "paraphrase_query": "analytics events for show and click", "scope": "project:online"},
    {"jargon_query": "переключение вкладок в карточке",
     "paraphrase_query": "switching tabs inside the entity card", "scope": "project:online"},
    {"jargon_query": "не коммитить промежуточные исследования",
     "paraphrase_query": "do not commit intermediate research artifacts", "scope": "project:online"},
    {"jargon_query": "проверка соответствия вёрстки макету",
     "paraphrase_query": "verifying design parity against the mockup", "scope": "project:online"},
    {"jargon_query": "как обрабатывать ошибки на слое данных",
     "paraphrase_query": "how errors are handled in the data layer", "scope": "project:online"},
    {"jargon_query": "почему всплывающая подсказка ломает карту",
     "paraphrase_query": "why the tooltip breaks map interaction stability", "scope": "project:online"},
    {"jargon_query": "почему макет и реализация разъезжаются по отступам",
     "paraphrase_query": "why the mockup and the implementation disagree on spacing",
     "scope": "project:online"},
    {"jargon_query": "переоткрыть цель через апи дашборда",
     "paraphrase_query": "how to reopen a goal through the dashboard API", "scope": "project:ae"},
    {"jargon_query": "воркtree ноды исполнения цели",
     "paraphrase_query": "what the node execution worktree is used for", "scope": "project:ae"},
    {"jargon_query": "почему узел провалился и как чинить",
     "paraphrase_query": "why a node failed and how it is repaired", "scope": "project:ae"},
    {"jargon_query": "как собирается контекст для промпта узла",
     "paraphrase_query": "how the node prompt context is assembled", "scope": "project:ae"},
    {"jargon_query": "запросы оператора в формате джейсон",
     "paraphrase_query": "operator requests in json format", "scope": "project:ae"},
    {"jargon_query": "почему приложение падает при отключении",
     "paraphrase_query": "why the agent process crashes on disconnect", "scope": "project:ae"},
    {"jargon_query": "как деревья целей ведут журнал решений",
     "paraphrase_query": "how goal trees record their decisions", "scope": "project:ae"},
    {"jargon_query": "как не потерять артефакт между воркtree",
     "paraphrase_query": "how not to lose an artifact between worktrees", "scope": "project:ae"},
    {"jargon_query": "смешанная точность обучения гейты",
     "paraphrase_query": "mixed precision training gates", "scope": "project:x"},
    {"jargon_query": "резидентность автограда на бэкенде",
     "paraphrase_query": "autograd engine device residency on the backend", "scope": "project:x"},
    {"jargon_query": "пластичное обновление скаляра куда",
     "paraphrase_query": "the cuda plastic scalar update path", "scope": "project:x"},
    {"jargon_query": "прожекции выровненных размерностей проверка",
     "paraphrase_query": "aligned dims projection verification", "scope": "project:x"},
    {"jargon_query": "игнорируемые артефакты не доезжают до родителя",
     "paraphrase_query": "ignored artifacts do not propagate to the parent worktree",
     "scope": "global"},
    {"jargon_query": "что считается регрессией на прогоне",
     "paraphrase_query": "what counts as a regression on a run", "scope": "global"},
)

#: ``role_query_rule``: role/procedural queries whose answer is an active
#: ``level:schema`` node carrying ``context.trigger``. ``anchor_node_id`` names
#: one copy; :func:`_expand_schema_copies` names every identical copy in the
#: same scope, so retrieval is not penalised for returning a twin. ``scope`` is
#: the procedure's own scope, so the item measures ranking rather than the scope
#: gate. Node ids were mined from the calibration snapshot's 433 active schema
#: nodes; the builder re-validates active + level + trigger for every one.
ROLE_QUERY_SPECS: tuple[dict[str, Any], ...] = (
    {"query": "как оформлять цель ревью джира тикета",
     "anchor_node_id": "01KZ9PG17MXWMPY25F8S7EH8T3", "scope": "global"},
    {"query": "какая моя роль когда оператор просит поревьювить тикет",
     "anchor_node_id": "01M07Q3MBKAEQ8SCT3E5G9KE1X", "scope": "global"},
    {"query": "как создавать или переоткрывать цель в дашборде",
     "anchor_node_id": "01KTSKWMQHD97Y6R7W6P4YCB3D", "scope": "global"},
    {"query": "проверить есть ли уже цель ревью по ключу и переоткрыть её",
     "anchor_node_id": "01KXQKC19BEEWM5559DASQZEY7", "scope": "project:online"},
    {"query": "что делать если мердж-реквест изменился после моего ревью",
     "anchor_node_id": "01KZXPK81DCQYTA6YPGSYYQTRM", "scope": "project:online"},
    {"query": "реоупен ревью говорит правки внесены с чего начать",
     "anchor_node_id": "01KTSB8N2E6HA4CSF2DXBQYK80", "scope": "project:online"},
    {"query": "как верифицировать фиксы на реоупене ревью читать дифф или исполнять",
     "anchor_node_id": "01KZDN9CBEGS3V494AH17PK3MF", "scope": "project:online"},
    {"query": "что фиксировать про head sha и версию диффа при публикации ревью",
     "anchor_node_id": "01KZDN9CBT96Q09V0CC20BPMS5", "scope": "project:online"},
    {"query": "автор пушит в ветку пока я ревьюю как тянуть дифф",
     "anchor_node_id": "01KVYS3SMN67XK6SPSHNQ9321B", "scope": "project:online"},
    {"query": "когда считать дизайн задачу выполненной по макетам",
     "anchor_node_id": "01KX5CXCRXZ4P84RTT6G09R40E", "scope": "project:online"},
    {"query": "какие файлы можно коммитить в ветку фичи",
     "anchor_node_id": "01KXQKC13VEE38PYHH5H2BSRV1", "scope": "project:online"},
    {"query": "что сверять перед завершением узла в ветке",
     "anchor_node_id": "01KZDN9CAGE14TE5SMR5SGW0A0", "scope": "project:online"},
    {"query": "как убрать отладочные артефакты из ветки перед результатом",
     "anchor_node_id": "01KZDN9CD0KE175DE6DSD7ZVC2", "scope": "project:online"},
    {"query": "куда класть вердикты и чек листы узлов цели",
     "anchor_node_id": "01KZDN9CCTFQ9F64A79QMDA8CK", "scope": "project:online"},
    {"query": "в реоупене дали ссылку на тред что она значит",
     "anchor_node_id": "01KVYS3SMQ8EEAQKD0GH9Z1BJ9", "scope": "project:online"},
    {"query": "как сверять урл пришедший query параметром",
     "anchor_node_id": "01KXQKC16H85527EZ7W674ZJYZ", "scope": "project:online"},
    {"query": "как точка входа на карте должна отправлять сигнал открытия",
     "anchor_node_id": "01KX5CXCSHKJBFYAK905VY3JPH", "scope": "project:online"},
    {"query": "как проверять запись булева значения в настройках",
     "anchor_node_id": "01KV5DJE2J3Y5X08K0PFGHA0NA", "scope": "project:online"},
    {"query": "карточка открыта из выдачи двигаем карту что происходит с поиском",
     "anchor_node_id": "01KV5DJDYPP51KZCCZMZ4Z888Y", "scope": "project:online"},
    {"query": "какие точки входа у пакета карты на десктопе",
     "anchor_node_id": "01KTCH9T0RKRAW05VWFE7BPPQW", "scope": "project:online"},
    {"query": "как устроен рестарт сервера памяти через админку",
     "anchor_node_id": "01KS2P8T67G5D0AQ9M6AQ50CNW", "scope": "project:lm"},
    {"query": "что соблюдать при правке инструкций сервера памяти",
     "anchor_node_id": "01KZW43085CNC4M71VWS8EAVJ4", "scope": "project:lm"},
    {"query": "как перегенерировать ядро релиза после валидатора",
     "anchor_node_id": "01M00P4SW0TEWDY9JZRSYAKWBA", "scope": "project:lm"},
    {"query": "как доказать из какой живой базы сделан снапшот",
     "anchor_node_id": "01M02HRXEVCYPFJMN91MQ559T2", "scope": "project:lm"},
    {"query": "как присматривать за активными целями и не мешать им",
     "anchor_node_id": "01KT4E4T0A4XD4QZZRV3FSV9NY", "scope": "project:ae"},
    {"query": "ложный алерт про схему как чинить",
     "anchor_node_id": "01KS57467A0N75QXYVPXA9WCN5", "scope": "project:ae"},
    {"query": "дифф трогает промпты что требует гейт от коммита",
     "anchor_node_id": "01KY5HQ9GPQ02PC3C03H5N6RWT", "scope": "project:ae"},
    {"query": "как авторизоваться в апи дашборда локально",
     "anchor_node_id": "01KT6GV0HQF5D3D4XJSXABF3B1", "scope": "project:ae"},
    {"query": "как показывать секреты в интерфейсе",
     "anchor_node_id": "01KZ8RYA4Q3Y3YMF9X7BG8HVR1", "scope": "project:ae"},
    {"query": "что подавать планировщику при досборке дерева",
     "anchor_node_id": "01KY5HQ9G26J8WTVQD69ENWCB7", "scope": "project:ae"},
    {"query": "ревью артефакта нашло дефекты что отдавать в результате",
     "anchor_node_id": "01KWQ5YJCYJ6QHXHVT4WX8TR8N", "scope": "project:ae"},
    {"query": "как устроены постоянные сессии чата в дашборде",
     "anchor_node_id": "01KY5HQ9F8MH1MYJMBF62HPBFA", "scope": "project:ae"},
    {"query": "что нужно знать про озвучку сообщений перед началом",
     "anchor_node_id": "01KWQ5YJBD6PYEDXWKNJFT21VQ", "scope": "project:ae"},
    {"query": "канарейка выдала подозрительно чистые числа что делать",
     "anchor_node_id": "01KRXTWDXPCQNSDZH1VVEZFF8D", "scope": "project:octopus"},
    {"query": "тесты пишут продакшн артефакты как чинить приёмку",
     "anchor_node_id": "01KTSKWPGKT0WDQWXE6J8DWXFS", "scope": "project:octopus"},
    {"query": "что блокирует декомпозицию на критике",
     "anchor_node_id": "01KSQ335AEA39CEWX5HHMVW0Q0", "scope": "project:octopus"},
    {"query": "перезапускать ли неудачный замер лестницы точности без изменений",
     "anchor_node_id": "01KTZ5GKADERTK4ZPD1NWCE2H9", "scope": "project:x"},
    {"query": "выбор преемника провалился как чинить родителя",
     "anchor_node_id": "01KW15GSTVDDHXHHSJBTPMXAKS", "scope": "project:x"},
    {"query": "какие режимы у раннера однократной оценки",
     "anchor_node_id": "01KVZZ99M6Q2ABAH64VFN5K90Q", "scope": "project:x"},
    {"query": "как атрибутировать счётчики телеметрии по владельцу",
     "anchor_node_id": "01KTS6WQNMZ4MEFW01FHCMYSWM", "scope": "project:x"},
    {"query": "что входит в заморозку исследовательского контракта",
     "anchor_node_id": "01KVRSYKRECF4XB2Q8YB2776KA", "scope": "project:x"},
    {"query": "как привязать предрегистрацию второй волны",
     "anchor_node_id": "01KVFRV21P0B5Q941F8T82HJQN", "scope": "project:x"},
    {"query": "как закрыть время жизни потока в оптимизаторе",
     "anchor_node_id": "01KTS6WQNS4F2X1GE1BDWBJBEX", "scope": "project:x"},
    {"query": "чинить два файла разными детьми или одним",
     "anchor_node_id": "01KTZ5GK8T331Q5ANEC2ZRWX2G", "scope": "project:x"},
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _normalized(text: str) -> str:
    return _WHITESPACE.sub(" ", str(text)).strip().casefold()


def _norm_hash(text: str) -> str:
    return hashlib.sha256(_normalized(text).encode("utf-8")).hexdigest()


def _latin_share(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    latin = sum(1 for char in letters if "LATIN" in unicodedata.name(char, ""))
    return latin / len(letters)


def _percentiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def at(fraction: float) -> float:
        if len(ordered) == 1:
            return round(ordered[0], 6)
        position = fraction * (len(ordered) - 1)
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return round(ordered[low], 6)
        weight = position - low
        return round(ordered[low] * (1 - weight) + ordered[high] * weight, 6)

    return {
        "min": round(ordered[0], 6),
        "p25": at(0.25),
        "p50": at(0.50),
        "p75": at(0.75),
        "p90": at(0.90),
        "max": round(ordered[-1], 6),
        "mean": round(statistics.fmean(ordered), 6),
        "n": len(ordered),
    }


@dataclass(frozen=True)
class RankedAnchor:
    """One anchor a calibration query sees, before any floor is applied."""

    anchor_id: str
    similarity: float
    targets: tuple[tuple[str, float], ...]


# ---------------------------------------------------------------------------
# Purity: exclusions derived from the scored goldsets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Exclusions:
    """Opaque keys of the scored items, for exclusion only.

    Deliberately holds no query text, no relevant node ids and no scores: the
    calibration must be able to prove it did not touch the answer key, and the
    cheapest proof is not carrying it.
    """

    sources: tuple[str, ...]
    fingerprints: frozenset[str]
    event_ids: frozenset[str]
    norm_hashes: frozenset[str]
    vectors: tuple[tuple[float, ...], ...]
    item_count: int

    def collides(self, query: str, scope: str | None, event_id: str | None) -> str | None:
        if event_id and str(event_id) in self.event_ids:
            return "source_event_id"
        if _norm_hash(query) in self.norm_hashes:
            return "normalized_query"
        if recall_fingerprint(query, scope or "global") in self.fingerprints:
            return "recall_fingerprint"
        return None

    def nearest(self, vector: Sequence[float]) -> float:
        return max((cosine_similarity(vector, other) for other in self.vectors), default=0.0)


def load_exclusions(paths: Sequence[Path], embed) -> Exclusions:
    """Read the scored goldsets and return exclusion keys, nothing else.

    The only values that leave this function are hashes, ids and vectors. The
    scored queries' text is embedded and discarded inside the loop; relevance
    labels and scores are never read off the record at all.
    """

    fingerprints: set[str] = set()
    event_ids: set[str] = set()
    norm_hashes: set[str] = set()
    vectors: list[tuple[float, ...]] = []
    total = 0
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            query = str(record.get("query") or "")
            if not query:
                continue
            total += 1
            scope = record.get("scope")
            norm_hashes.add(_norm_hash(query))
            fingerprints.add(recall_fingerprint(query, scope or "global"))
            # Both the item's own scope and the bare global key: an anchor
            # fingerprint is scope-qualified, and a calibration item may name
            # the same question in a different scope.
            fingerprints.add(recall_fingerprint(query, "global"))
            source_event_id = record.get("source_event_id")
            if source_event_id:
                event_ids.add(str(source_event_id))
            vectors.append(tuple(embed(query)))
    return Exclusions(
        sources=tuple(str(path) for path in paths),
        fingerprints=frozenset(fingerprints),
        event_ids=frozenset(event_ids),
        norm_hashes=frozenset(norm_hashes),
        vectors=tuple(vectors),
        item_count=total,
    )


# ---------------------------------------------------------------------------
# Calibration set construction
# ---------------------------------------------------------------------------


def _expand_schema_copies(store: MemoryStore, node_id: str) -> list[str]:
    """Every active schema node in the same scope with identical content.

    ``role_query_rule``: "where a procedure exists as several identical copies,
    ALL copies are named so retrieval is not penalised for returning a twin".
    Applied mechanically here rather than transcribed into the spec.
    """

    node = store.get_node(node_id)
    if node is None:
        raise SystemExit(f"role_query spec names unknown node {node_id}")
    digest = hashlib.sha1(node.content.encode("utf-8")).hexdigest()
    rows = store.connection.execute(
        "SELECT id, content FROM nodes WHERE level='schema' AND decayed=0 AND scope=?",
        (node.scope,),
    ).fetchall()
    copies = [
        str(row["id"])
        for row in rows
        if hashlib.sha1(str(row["content"]).encode("utf-8")).hexdigest() == digest
    ]
    return sorted(set(copies) | {node_id})


def _build_content_grounded(
    store: MemoryStore,
    tokenizer: Any,
    config: GoldsetBuildConfig,
    window_end: str,
    cap: int,
    seed: int,
    build_stamp: Mapping[str, Any],
) -> tuple[list[GoldsetItem], dict[str, Any]]:
    """The shipped builder, then the upper bound this split adds.

    ``build_content_grounded`` selects ``created_at >= cutoff`` with no upper
    bound, so the window's closing edge is applied here. Nothing about the
    label changes: the containment verdict, the active-node filter and the
    provenance are the shipped ones, which is the point -- a calibration item
    must mean the same thing a scored ``content_grounded`` item means.
    """

    items, stats = build_content_grounded(
        store.connection, config, tokenizer=tokenizer, build_stamp=build_stamp
    )
    end = normalize_cutoff(window_end)
    in_window = [
        item
        for item in items
        if str(item.provenance.get("recorded_created_at") or "") < end
    ]
    ordered = sorted(in_window, key=lambda item: item.query_id)
    if 0 <= cap < len(ordered):
        chosen = random.Random(seed).sample(range(len(ordered)), cap)
        ordered = [ordered[index] for index in sorted(chosen)]
    stats = dict(stats)
    stats.update(
        {
            "window_start": config.cutoff,
            "window_end": end,
            "at_or_after_window_start": len(items),
            "inside_window": len(in_window),
            "cap": cap,
            "sampled": len(ordered),
            "sampling": f"random.Random({seed}).sample over query_id-sorted in-window items",
        }
    )
    return ordered, stats


def _filter_cross_lingual(
    service: MemoryRecallService, items: Sequence[GoldsetItem]
) -> tuple[list[GoldsetItem], list[dict[str, Any]]]:
    """Re-verify ``cross_lingual_rule`` (a)-(c) mechanically on this snapshot.

    (d), "the node is topically about the query's subject", is an authoring
    judgement and is recorded as such -- 151 pairs were authored, measured, and
    read; the ones whose resolved node was off-topic were removed from
    :data:`CROSS_LINGUAL_SPECS` rather than being carried here with a flag.

    Node-level embeddings are empty on a chunked corpus, so the jargon query's
    score against the resolved node is its best chunk cosine -- the same
    quantity ``_collect_vector`` ranks on.
    """

    kept: list[GoldsetItem] = []
    audit: list[dict[str, Any]] = []
    for item in items:
        provenance = item.provenance
        top = (provenance.get("paraphrase_top_k") or [{}])[0]
        node = service.store.get_node(item.relevant_node_ids[0])
        if node is None:
            audit.append({"query_id": item.query_id, "verdict": "node_missing"})
            continue
        jargon_vector = service.embedder.embed(item.query)
        chunk_vectors = [
            chunk.embedding
            for chunk in service.store.list_node_chunks(node.id)
            if chunk.embedding
        ]
        if node.embedding:
            chunk_vectors.append(node.embedding)
        jargon_score = max(
            (cosine_similarity(jargon_vector, vector) for vector in chunk_vectors),
            default=None,
        )
        jargon_results = service.memory_recall(
            item.query,
            scope=item.scope,
            ambient_context=None,
            depth=item.depth,
            max_results=1,
            log_access=False,
            log_event=False,
        )
        ranks_first = bool(jargon_results) and jargon_results[0].node_id == node.id
        paraphrase_vector = float(top.get("vector_score") or 0.0)
        latin = _latin_share(node.content)
        checks = {
            "paraphrase_vector_at_least_0_40": paraphrase_vector >= 0.40,
            "node_predominantly_latin": latin >= 0.60,
            "jargon_scores_strictly_lower": (
                jargon_score is not None and jargon_score < paraphrase_vector
            ),
            "jargon_does_not_rank_it_first": not ranks_first,
        }
        record = {
            "query_id": item.query_id,
            "scope": item.scope,
            "relevant_node_id": node.id,
            "paraphrase_vector": round(paraphrase_vector, 6),
            "jargon_node_cosine": None if jargon_score is None else round(jargon_score, 6),
            "latin_share": round(latin, 4),
            "jargon_ranks_it_first": ranks_first,
            "checks": checks,
            "kept": all(checks.values()),
        }
        audit.append(record)
        if record["kept"]:
            kept.append(
                replace(
                    item,
                    provenance={
                        **provenance,
                        "calibration_rule_checks": checks,
                        "jargon_node_cosine": record["jargon_node_cosine"],
                        "node_latin_share": record["latin_share"],
                        "topicality": "author_verified_at_spec_time",
                    },
                )
            )
    return kept, audit


def _apply_exclusions(
    items: Sequence[GoldsetItem], exclusions: Exclusions, embed, near_dup: float
) -> tuple[list[GoldsetItem], dict[str, Any]]:
    """Drop every calibration item that touches the scored set. One-way."""

    kept: list[GoldsetItem] = []
    dropped: list[dict[str, Any]] = []
    nearest_all: list[float] = []
    nearest_kept: dict[str, list[float]] = defaultdict(list)
    for item in items:
        reason = exclusions.collides(item.query, item.scope, item.source_event_id)
        nearest = exclusions.nearest(embed(item.query))
        nearest_all.append(nearest)
        if reason is None and nearest >= near_dup:
            reason = "near_duplicate_query"
        if reason is None:
            nearest_kept[item.stratum].append(nearest)
        if reason is not None:
            dropped.append(
                {
                    "query_id": item.query_id,
                    "stratum": item.stratum,
                    "reason": reason,
                    "nearest_scored_cosine": round(nearest, 6),
                }
            )
            continue
        kept.append(item)
    report = {
        "scored_sources": list(exclusions.sources),
        "scored_items_scanned": exclusions.item_count,
        "near_dup_cosine": near_dup,
        "examined": len(items),
        "dropped": len(dropped),
        "dropped_detail": dropped,
        "kept": len(kept),
        "nearest_scored_cosine_examined": _percentiles(nearest_all),
        "nearest_scored_cosine_kept": {
            stratum: _percentiles(values) for stratum, values in sorted(nearest_kept.items())
        }
        | {
            "all": _percentiles(
                [value for values in nearest_kept.values() for value in values]
            )
        },
        "note": (
            "The scored goldsets are opened only to exclude. No scored query "
            "text, relevance label or score is read, stored or reported here."
        ),
    }
    return kept, report


def cmd_build_set(args: argparse.Namespace) -> int:
    working = Path(args.working_db)
    if not working.exists() or args.refresh_working:
        backup_database(args.snapshot, working)
    store = MemoryStore(MemoryConfig(db_path=working))
    try:
        service = MemoryRecallService(store, anchor_seeding=False)
        tokenizer = ModelVisibleSpan()
        build_stamp = {
            "builder": "scripts/anchor_match_calibration.py",
            "snapshot_sha256": sha256_file(args.snapshot),
            "window_start": normalize_cutoff(args.window_start),
            "window_end": normalize_cutoff(args.window_end),
            "seed": args.seed,
        }
        config = GoldsetBuildConfig(
            cutoff=args.window_start,
            seed=args.seed,
            content_grounded_cap=None,
            cross_lingual_cap=None,
            role_query_cap=None,
            # The recorded rule resolves relevance as the paraphrase's single
            # top result, not its top five.
            cross_lingual_top_k=1,
        )

        grounded, grounded_stats = _build_content_grounded(
            store,
            tokenizer,
            config,
            args.window_end,
            args.max_content_grounded,
            args.seed,
            build_stamp,
        )
        cross_raw, cross_stats = build_cross_lingual(
            service, CROSS_LINGUAL_SPECS, config, tokenizer=tokenizer, build_stamp=build_stamp
        )
        cross, cross_audit = _filter_cross_lingual(service, cross_raw)
        role_specs = [
            {**spec, "expected_node_ids": _expand_schema_copies(store, spec["anchor_node_id"])}
            for spec in ROLE_QUERY_SPECS
        ]
        role, role_stats = build_role_query(
            store, role_specs, config, tokenizer=tokenizer, build_stamp=build_stamp
        )

        candidates = list(grounded) + list(cross) + list(role)
        exclusions = load_exclusions(
            [Path(path) for path in args.scored_goldset], service.embedder.embed
        )
        items, purity = _apply_exclusions(
            candidates, exclusions, service.embedder.embed, args.near_dup_cosine
        )
        temporal = _temporal_assertions(items, args.window_start, args.window_end)
    finally:
        store.close()

    items.sort(key=lambda item: (item.stratum, item.query_id))
    dump_goldset(items, args.out)
    strata: dict[str, int] = defaultdict(int)
    for item in items:
        strata[item.stratum] += 1
    report = {
        "artifact": str(args.out),
        "snapshot": str(args.snapshot),
        "snapshot_sha256": build_stamp["snapshot_sha256"],
        "window": {"start": build_stamp["window_start"], "end": build_stamp["window_end"]},
        "seed": args.seed,
        "items": len(items),
        "per_stratum": dict(sorted(strata.items())),
        "content_grounded": grounded_stats,
        "cross_lingual": {
            **cross_stats,
            "authored_specs": len(CROSS_LINGUAL_SPECS),
            "kept_after_rule_recheck": len(cross),
            "rule_audit": cross_audit,
        },
        "role_query": {**role_stats, "authored_specs": len(ROLE_QUERY_SPECS)},
        "purity": purity,
        "temporal": temporal,
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"calibration set: {len(items)} items {dict(sorted(strata.items()))} "
        f"-> {args.out} (report {args.report})",
        file=sys.stderr,
    )
    return 0


def _temporal_assertions(
    items: Sequence[GoldsetItem], window_start: str, window_end: str
) -> dict[str, Any]:
    """Every timestamped item sits inside the window, strictly before the scored cutoff."""

    start = normalize_cutoff(window_start)
    end = normalize_cutoff(window_end)
    stamps = [
        str(item.provenance.get("recorded_created_at") or "")
        for item in items
        if item.source_event_id
    ]
    outside = [stamp for stamp in stamps if not (start <= stamp < end)]
    return {
        "window_start": start,
        "window_end": end,
        "timestamped_items": len(stamps),
        "untimestamped_items": len(items) - len(stamps),
        "earliest": min(stamps) if stamps else None,
        "latest": max(stamps) if stamps else None,
        "outside_window": len(outside),
        "ok": not outside,
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def _rank_all(service: MemoryRecallService, items: Sequence[GoldsetItem]) -> dict[str, list[RankedAnchor]]:
    """Every item's nearest anchors once, unfloored.

    One pass instead of one scan per swept cell: a (threshold, limit) pair only
    ever selects a prefix of this list, so coverage, precision and the seed set
    of every cell are derivable from it exactly.
    """

    source = service._anchor_vector_source()
    ranked: dict[str, list[RankedAnchor]] = {}
    for item in items:
        if source is None:
            ranked[item.query_id] = []
            continue
        plan = service.scope_resolver.resolve(
            query=item.query,
            scope=item.scope,
            ambient_context=item.ambient_context,
            store=service.store,
        )
        vector = service.embedder.embed(item.query)
        matches = match_anchors(
            source, vector, plan, limit=RANK_DEPTH, min_similarity=-1.0
        )
        ranked[item.query_id] = [
            RankedAnchor(match.anchor.id, float(match.similarity), tuple(match.targets))
            for match in matches
        ]
    return ranked


def _seeds_for(anchors: Sequence[RankedAnchor]) -> dict[str, float]:
    """``_collect_anchor_seeds``' arithmetic, reproduced on ranked anchors."""

    seeds: dict[str, float] = {}
    for anchor in anchors:
        for target_id, weight in anchor.targets:
            activation = anchor.similarity * min(1.0, max(0.0, weight))
            if activation > seeds.get(target_id, 0.0):
                seeds[target_id] = activation
    if len(seeds) <= GRAPH_SEED_LIMIT:
        return seeds
    return dict(
        sorted(seeds.items(), key=lambda entry: (-entry[1], entry[0]))[:GRAPH_SEED_LIMIT]
    )


def _seed_signature(seeds: Mapping[str, float]) -> str:
    payload = ";".join(f"{node}:{value:.9f}" for node, value in sorted(seeds.items()))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _metrics(order: Sequence[str], relevant: Iterable[str]) -> dict[str, float]:
    relevant_set = set(relevant)
    rank = next(
        (index + 1 for index, node_id in enumerate(order) if node_id in relevant_set), None
    )
    return {
        "hit@1": 1.0 if rank == 1 else 0.0,
        "hit@5": 1.0 if rank is not None and rank <= 5 else 0.0,
        "hit@10": 1.0 if rank is not None and rank <= 10 else 0.0,
        "mrr": 0.0 if rank is None else 1.0 / rank,
    }


def _aggregate(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {"items": 0, "hit@1": 0.0, "hit@5": 0.0, "hit@10": 0.0, "mrr": 0.0}
    return {
        "items": len(rows),
        **{
            key: round(statistics.fmean([row[key] for row in rows]), 6)
            for key in ("hit@1", "hit@5", "hit@10", "mrr")
        },
    }


def _run_item(service: MemoryRecallService, item: GoldsetItem) -> tuple[list[str], float]:
    started = time.perf_counter()
    results = service.memory_recall(
        item.query,
        scope=item.scope,
        ambient_context=item.ambient_context,
        depth=item.depth,
        max_results=item.max_results,
        log_access=False,
        log_event=False,
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    return [result.node_id for result in results], elapsed


def cmd_sweep(args: argparse.Namespace) -> int:
    items = load_goldset(args.calibration_set)
    working = Path(args.working_db)
    if not working.exists() or args.refresh_working:
        backup_database(args.snapshot, working)
    thresholds = tuple(args.threshold) or DEFAULT_THRESHOLDS
    limits = tuple(args.limit) or DEFAULT_LIMITS

    store = MemoryStore(MemoryConfig(db_path=working))
    try:
        service = MemoryRecallService(store, anchor_seeding=True)
        corpus = {
            "anchors": store.count_query_anchors(),
            "live_anchors": store.count_query_anchors(include_decayed=False),
            "edges": store.count_query_anchor_edges(),
        }
        print(f"anchor corpus: {corpus}", file=sys.stderr)
        ranked = _rank_all(service, items)

        # Arm A: anchors off. Also the fallback ranking of every cell in which
        # an item matches nothing -- with no seed the anchor code path is inert,
        # which ``--verify-inert`` re-proves by running it for real.
        service.anchor_seeding = False
        baseline_order: dict[str, list[str]] = {}
        baseline_ms: dict[str, float] = {}
        for index, item in enumerate(items, 1):
            order, elapsed = _run_item(service, item)
            baseline_order[item.query_id] = order
            baseline_ms[item.query_id] = elapsed
            if index % 50 == 0:
                print(f"  baseline {index}/{len(items)}", file=sys.stderr)
        service.anchor_seeding = True

        by_id = {item.query_id: item for item in items}
        cache: dict[tuple[str, str], tuple[list[str], float]] = {}
        empty_signature = _seed_signature({})
        for query_id, order in baseline_order.items():
            cache[(query_id, empty_signature)] = (order, baseline_ms[query_id])

        cells: list[dict[str, Any]] = []
        for threshold in thresholds:
            for limit in limits:
                cells.append(
                    _sweep_cell(
                        service,
                        by_id,
                        items,
                        ranked,
                        threshold,
                        limit,
                        cache,
                        empty_signature,
                        baseline_order,
                        baseline_ms,
                    )
                )
                cell = cells[-1]
                print(
                    f"  t={threshold:.2f} L={limit}: coverage {cell['coverage']['share']:.3f} "
                    f"anchor-precision {cell['precision']['anchor_precision']} "
                    f"item-precision {cell['precision']['item_precision']} "
                    f"changed {cell['quality']['ranking_changed_items']} "
                    f"(+{cell['quality']['items_improved']}/-{cell['quality']['items_regressed']}) "
                    f"hit@5 {cell['quality']['overall']['hit@5']:.4f} "
                    f"mrr {cell['quality']['overall']['mrr']:.4f}",
                    file=sys.stderr,
                )

        inert = _verify_inert(service, by_id, ranked, baseline_order, args.verify_inert)
    finally:
        store.close()

    baseline_rows = [
        _metrics(baseline_order[item.query_id], item.relevant_node_ids) for item in items
    ]
    baseline_by_stratum: dict[str, list[dict[str, float]]] = defaultdict(list)
    for item, row in zip(items, baseline_rows):
        baseline_by_stratum[item.stratum].append(row)

    report = {
        "artifact": str(args.out_json),
        "what": (
            "Calibration of ANCHOR_MATCH_COSINE_THRESHOLD / ANCHOR_MATCH_LIMIT on a "
            "calibration set built strictly inside the scored run's anchor training "
            "window, never on the scored goldset."
        ),
        "snapshot": str(args.snapshot),
        "snapshot_sha256": sha256_file(args.snapshot),
        "calibration_set": str(args.calibration_set),
        "calibration_set_sha256": sha256_file(args.calibration_set),
        "anchor_corpus": corpus,
        "items": len(items),
        "per_stratum_items": {
            stratum: len(rows) for stratum, rows in sorted(baseline_by_stratum.items())
        },
        "nearest_anchor_cosine": _nearest_summary(items, ranked),
        "baseline_no_anchors": {
            "overall": _aggregate(baseline_rows),
            "per_stratum": {
                stratum: _aggregate(rows) for stratum, rows in sorted(baseline_by_stratum.items())
            },
            "recall_ms": _percentiles(list(baseline_ms.values())),
        },
        "grid": {"thresholds": list(thresholds), "limits": list(limits)},
        "cells": cells,
        "inert_path_check": inert,
        "latency_model": {
            "source": "artifacts/anchors/latency.json",
            "matched_paired_p50_ms": LATENCY_MATCHED_P50_MS,
            "unmatched_paired_p50_ms": LATENCY_UNMATCHED_P50_MS,
            "budget_ms": LATENCY_BUDGET_MS,
            "rule": (
                "projected p50 paired delta = the median of the mixture the measured "
                "match rate implies: the matched median once the matched share exceeds "
                "one half, the unmatched median below it, interpolated between."
            ),
        },
    }
    Path(args.out_json).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.out_json}", file=sys.stderr)
    return 0


def _nearest_summary(
    items: Sequence[GoldsetItem], ranked: Mapping[str, Sequence[RankedAnchor]]
) -> dict[str, Any]:
    per_stratum: dict[str, list[float]] = defaultdict(list)
    pool: dict[str, list[int]] = defaultdict(list)
    for item in items:
        anchors = ranked.get(item.query_id) or []
        per_stratum[item.stratum].append(anchors[0].similarity if anchors else 0.0)
        pool[item.stratum].append(len(anchors))
    summary = {
        stratum: _percentiles(values) for stratum, values in sorted(per_stratum.items())
    }
    summary["all"] = _percentiles([value for values in per_stratum.values() for value in values])
    summary["ranked_depth_note"] = (
        f"pool sizes are truncated at RANK_DEPTH={RANK_DEPTH}; "
        f"queries with a full list: "
        f"{sum(1 for values in pool.values() for value in values if value >= RANK_DEPTH)}"
    )
    return summary


def _sweep_cell(
    service: MemoryRecallService,
    by_id: Mapping[str, GoldsetItem],
    items: Sequence[GoldsetItem],
    ranked: Mapping[str, Sequence[RankedAnchor]],
    threshold: float,
    limit: int,
    cache: dict[tuple[str, str], tuple[list[str], float]],
    empty_signature: str,
    baseline_order: Mapping[str, Sequence[str]],
    baseline_ms: Mapping[str, float],
) -> dict[str, Any]:
    """One (threshold, limit) cell: coverage, precision, quality, cost."""

    retrieval_module.ANCHOR_MATCH_COSINE_THRESHOLD = threshold
    retrieval_module.ANCHOR_MATCH_LIMIT = limit

    matched_items = 0
    matched_anchors = 0
    precise_anchors = 0
    useful_matched_items = 0
    seeded_total = 0
    relevant_seeds = 0
    seeds_per_matched: list[int] = []
    per_stratum_matched: dict[str, int] = defaultdict(int)
    per_stratum_useful: dict[str, int] = defaultdict(int)
    per_stratum_items: dict[str, int] = defaultdict(int)
    per_stratum_anchor_hits: dict[str, list[int]] = defaultdict(list)
    rows: list[dict[str, float]] = []
    rows_by_stratum: dict[str, list[dict[str, float]]] = defaultdict(list)
    durations: list[float] = []
    matched_deltas: list[float] = []
    changed = 0
    improved: list[str] = []
    regressed: list[str] = []

    for item in items:
        per_stratum_items[item.stratum] += 1
        selected = [
            anchor
            for anchor in (ranked.get(item.query_id) or [])
            if anchor.similarity >= threshold
        ][:limit]
        relevant = set(item.relevant_node_ids)
        if selected:
            matched_items += 1
            per_stratum_matched[item.stratum] += 1
        useful = False
        for anchor in selected:
            matched_anchors += 1
            hit = any(target_id in relevant for target_id, _ in anchor.targets)
            useful = useful or hit
            precise_anchors += int(hit)
            per_stratum_anchor_hits[item.stratum].append(int(hit))
        if useful:
            useful_matched_items += 1
            per_stratum_useful[item.stratum] += 1

        seeds = _seeds_for(selected)
        seeded_total += len(seeds)
        relevant_seeds += sum(1 for node_id in seeds if node_id in relevant)
        if selected:
            seeds_per_matched.append(len(seeds))

        signature = _seed_signature(seeds)
        key = (item.query_id, signature)
        if key not in cache:
            cache[key] = _run_item(service, item)
        order, elapsed = cache[key]
        durations.append(elapsed)
        if signature != empty_signature:
            matched_deltas.append(elapsed - baseline_ms[item.query_id])
        base = list(baseline_order[item.query_id])
        if order != base:
            changed += 1
            before = _metrics(base, relevant)["mrr"]
            after = _metrics(order, relevant)["mrr"]
            if after > before:
                improved.append(item.query_id)
            elif after < before:
                regressed.append(item.query_id)
        row = _metrics(order, relevant)
        rows.append(row)
        rows_by_stratum[item.stratum].append(row)

    share = matched_items / len(items) if items else 0.0
    return {
        "threshold": threshold,
        "limit": limit,
        "coverage": {
            "matched_items": matched_items,
            "items": len(items),
            "share": round(share, 6),
            "per_stratum": {
                stratum: {
                    "matched": per_stratum_matched.get(stratum, 0),
                    "items": count,
                    "share": round(per_stratum_matched.get(stratum, 0) / count, 6),
                }
                for stratum, count in sorted(per_stratum_items.items())
            },
        },
        "precision": {
            "matched_anchors": matched_anchors,
            "anchors_with_relevant_edge": precise_anchors,
            "anchor_precision": (
                round(precise_anchors / matched_anchors, 6) if matched_anchors else None
            ),
            "matched_items_with_a_relevant_edge": useful_matched_items,
            "item_precision": (
                round(useful_matched_items / matched_items, 6) if matched_items else None
            ),
            "seeded_nodes": seeded_total,
            "relevant_seeded_nodes": relevant_seeds,
            "seed_precision": (
                round(relevant_seeds / seeded_total, 6) if seeded_total else None
            ),
            "per_stratum_anchor_precision": {
                stratum: round(statistics.fmean(hits), 6) if hits else None
                for stratum, hits in sorted(per_stratum_anchor_hits.items())
            },
            "per_stratum_item_precision": {
                stratum: (
                    round(per_stratum_useful.get(stratum, 0) / per_stratum_matched[stratum], 6)
                    if per_stratum_matched.get(stratum)
                    else None
                )
                for stratum in sorted(per_stratum_items)
            },
        },
        "quality": {
            "overall": _aggregate(rows),
            "per_stratum": {
                stratum: _aggregate(stratum_rows)
                for stratum, stratum_rows in sorted(rows_by_stratum.items())
            },
            "ranking_changed_items": changed,
            "items_improved": len(improved),
            "items_regressed": len(regressed),
            "improved_ids": improved,
            "regressed_ids": regressed,
        },
        "cost": {
            "matched_share": round(share, 6),
            "seeds_per_matched_query": _percentiles([float(v) for v in seeds_per_matched]),
            "seeded_nodes_total": seeded_total,
            "measured_recall_ms": _percentiles(durations),
            "single_shot_paired_delta_ms_matched": _percentiles(matched_deltas),
            "projected_paired_p50_delta_ms": round(_projected_delta(share), 4),
            "within_budget": _projected_delta(share) <= LATENCY_BUDGET_MS,
        },
    }


def _projected_delta(matched_share: float) -> float:
    """The p50 of the paired-delta mixture a match rate implies.

    Below a 50% match rate the median paired query is an unmatched one, so the
    p50 is the unmatched median; above it, the matched median. In between the
    p50 crosses, and a linear interpolation across the crossing is the honest
    reading of two measured points -- it is a projection from
    ``artifacts/anchors/latency.json``, not a new measurement, and each cell
    publishes its own measured wall time beside it.
    """

    share = min(1.0, max(0.0, matched_share))
    if share <= 0.4:
        return LATENCY_UNMATCHED_P50_MS
    if share >= 0.6:
        return LATENCY_MATCHED_P50_MS
    weight = (share - 0.4) / 0.2
    return LATENCY_UNMATCHED_P50_MS + weight * (
        LATENCY_MATCHED_P50_MS - LATENCY_UNMATCHED_P50_MS
    )


def _verify_inert(
    service: MemoryRecallService,
    by_id: Mapping[str, GoldsetItem],
    ranked: Mapping[str, Sequence[RankedAnchor]],
    baseline_order: Mapping[str, list[str]],
    sample: int,
) -> dict[str, Any]:
    """Prove the reuse the sweep depends on: no seed means the baseline ranking.

    Every cell reuses the anchors-off ranking for an item that matched nothing,
    on the argument that ``_collect_anchor_seeds`` returning ``{}`` leaves the
    code path inert. That argument is cheap to check and expensive to be wrong
    about, so it is checked: a sample of items is re-run for real with anchors
    ON and an unreachable floor, and the ranking must come back identical.
    """

    if sample <= 0:
        return {"checked": 0, "identical": 0, "ok": True, "skipped": True}
    retrieval_module.ANCHOR_MATCH_COSINE_THRESHOLD = 1.01
    retrieval_module.ANCHOR_MATCH_LIMIT = 5
    query_ids = sorted(baseline_order)
    step = max(1, len(query_ids) // sample)
    checked = 0
    identical = 0
    divergent: list[str] = []
    for query_id in query_ids[::step][:sample]:
        order, _ = _run_item(service, by_id[query_id])
        checked += 1
        if order == baseline_order[query_id]:
            identical += 1
        else:
            divergent.append(query_id)
    return {
        "checked": checked,
        "identical": identical,
        "divergent": divergent,
        "ok": not divergent,
        "method": "anchors ON at an unreachable floor (1.01) must equal the anchors-off arm",
    }


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Where a matched anchor's claim dies, item by item.

    The sweep shows quality flat at every floor while coverage moves by two
    orders of magnitude. Two very different worlds produce that: the matched
    anchors may point at nothing the baseline was missing (then the floor is
    irrelevant because the *edges* are), or they may point at exactly the right
    node and the ranker may refuse to surface it (then the floor is irrelevant
    because the *scoring* is). This separates them, so the chosen floor is
    published next to the reason it cannot show up as hit@5.

    The funnel, per item, at one (threshold, limit):

    ``matched`` -> ``has an edge to a relevant node`` -> ``the baseline missed
    that node at 5`` -> ``anchors-on now ranks it at 5``.
    """

    items = load_goldset(args.calibration_set)
    working = Path(args.working_db)
    if not working.exists() or args.refresh_working:
        backup_database(args.snapshot, working)
    store = MemoryStore(MemoryConfig(db_path=working))
    try:
        service = MemoryRecallService(store, anchor_seeding=True)
        ranked = _rank_all(service, items)
        retrieval_module.ANCHOR_MATCH_COSINE_THRESHOLD = args.threshold
        retrieval_module.ANCHOR_MATCH_LIMIT = args.limit

        funnel = {
            "items": len(items),
            "matched": 0,
            "matched_with_relevant_edge": 0,
            "relevant_edge_and_baseline_missed_at_5": 0,
            "rescued_into_top_5": 0,
            "relevant_edge_and_baseline_missed_at_10": 0,
            "rescued_into_top_10": 0,
        }
        per_stratum: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        detail: list[dict[str, Any]] = []
        for item in items:
            selected = [
                anchor
                for anchor in (ranked.get(item.query_id) or [])
                if anchor.similarity >= args.threshold
            ][: args.limit]
            if not selected:
                continue
            funnel["matched"] += 1
            per_stratum[item.stratum]["matched"] += 1
            relevant = set(item.relevant_node_ids)
            seeds = _seeds_for(selected)
            relevant_seeds = sorted(node_id for node_id in seeds if node_id in relevant)
            if not relevant_seeds:
                continue
            funnel["matched_with_relevant_edge"] += 1
            per_stratum[item.stratum]["matched_with_relevant_edge"] += 1

            service.anchor_seeding = False
            base_order, _ = _run_item(service, item)
            service.anchor_seeding = True
            on_order, _ = _run_item(service, item)

            base_rank = next(
                (i + 1 for i, node in enumerate(base_order) if node in relevant), None
            )
            on_rank = next(
                (i + 1 for i, node in enumerate(on_order) if node in relevant), None
            )
            seed_rank = next(
                (i + 1 for i, node in enumerate(on_order) if node in relevant_seeds), None
            )
            record = {
                "query_id": item.query_id,
                "stratum": item.stratum,
                "scope": item.scope,
                "matched_anchors": len(selected),
                "best_similarity": round(selected[0].similarity, 6),
                "seeds": len(seeds),
                "relevant_seeds": len(relevant_seeds),
                "relevant_seed_activation": round(
                    max(seeds[node_id] for node_id in relevant_seeds), 6
                ),
                "baseline_rank": base_rank,
                "anchors_on_rank": on_rank,
                "seeded_relevant_node_rank": seed_rank,
                "returned": len(on_order),
                "ranking_changed": on_order != base_order,
            }
            detail.append(record)
            missed5 = base_rank is None or base_rank > 5
            missed10 = base_rank is None or base_rank > 10
            if missed5:
                funnel["relevant_edge_and_baseline_missed_at_5"] += 1
                per_stratum[item.stratum]["relevant_edge_and_baseline_missed_at_5"] += 1
                if on_rank is not None and on_rank <= 5:
                    funnel["rescued_into_top_5"] += 1
                    per_stratum[item.stratum]["rescued_into_top_5"] += 1
            if missed10:
                funnel["relevant_edge_and_baseline_missed_at_10"] += 1
                if on_rank is not None and on_rank <= 10:
                    funnel["rescued_into_top_10"] += 1
    finally:
        store.close()

    report = {
        "artifact": str(args.out_json),
        "what": "Per-item funnel from a matched anchor to a rank change.",
        "threshold": args.threshold,
        "limit": args.limit,
        "calibration_set": str(args.calibration_set),
        "funnel": funnel,
        "per_stratum": {
            stratum: dict(counts) for stratum, counts in sorted(per_stratum.items())
        },
        "items_with_a_relevant_edge": detail,
        "reading": (
            "'matched_with_relevant_edge' minus 'relevant_edge_and_baseline_missed_at_5' "
            "is the population the entry cannot help because the baseline already had "
            "it. What remains is the population it could help, and "
            "'rescued_into_top_5' is how much of that it did."
        ),
    }
    Path(args.out_json).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(funnel, indent=1), file=sys.stderr)
    print(f"wrote {args.out_json}", file=sys.stderr)
    return 0


def cmd_latency(args: argparse.Namespace) -> int:
    """Paired recall latency at each candidate floor, arms alternated.

    ``artifacts/anchors/latency.json`` measured the matched stratum on **two**
    queries. A floor is chosen partly on what it costs, so two samples is not
    enough to choose on; this re-measures the same paired design on every
    calibration query that the floor actually matches.

    The design is copied deliberately: per-query medians differenced (per-query
    spread of 60-1100 ms dwarfs the effect), arms alternated every iteration so
    neither inherits the other's warm page cache, and reads that neither log an
    access nor an event so no arm mutates the copy.
    """

    items = load_goldset(args.calibration_set)
    working = Path(args.working_db)
    if not working.exists() or args.refresh_working:
        backup_database(args.snapshot, working)
    store = MemoryStore(MemoryConfig(db_path=working))
    measurements: list[dict[str, Any]] = []
    try:
        service = MemoryRecallService(store, anchor_seeding=True)
        ranked = _rank_all(service, items)
        for threshold in args.threshold:
            retrieval_module.ANCHOR_MATCH_COSINE_THRESHOLD = threshold
            retrieval_module.ANCHOR_MATCH_LIMIT = args.limit
            matched = [
                item
                for item in items
                if any(
                    anchor.similarity >= threshold
                    for anchor in (ranked.get(item.query_id) or [])
                )
            ]
            unmatched = [item for item in items if item not in matched]
            sample = _stride(matched, args.sample) + _stride(unmatched, args.sample)
            rows: list[dict[str, Any]] = []
            for item in sample:
                is_matched = item in matched
                with_ms: list[float] = []
                without_ms: list[float] = []
                for iteration in range(args.warmup + args.iters):
                    for arm in ((True, False) if iteration % 2 == 0 else (False, True)):
                        service.anchor_seeding = arm
                        _, elapsed = _run_item(service, item)
                        if iteration >= args.warmup:
                            (with_ms if arm else without_ms).append(elapsed)
                service.anchor_seeding = True
                rows.append(
                    {
                        "query_id": item.query_id,
                        "stratum": item.stratum,
                        "matched": is_matched,
                        "with_anchors_median_ms": round(statistics.median(with_ms), 4),
                        "without_anchors_median_ms": round(statistics.median(without_ms), 4),
                        "paired_delta_ms": round(
                            statistics.median(with_ms) - statistics.median(without_ms), 4
                        ),
                    }
                )
                print(
                    f"  t={threshold:.2f} {item.query_id} matched={is_matched} "
                    f"delta={rows[-1]['paired_delta_ms']:+.3f} ms",
                    file=sys.stderr,
                )
            deltas_matched = [row["paired_delta_ms"] for row in rows if row["matched"]]
            deltas_unmatched = [row["paired_delta_ms"] for row in rows if not row["matched"]]
            share = len(matched) / len(items) if items else 0.0
            pooled = _mixture_p50(deltas_matched, deltas_unmatched, share)
            measurements.append(
                {
                    "threshold": threshold,
                    "limit": args.limit,
                    "matched_items_in_set": len(matched),
                    "matched_share": round(share, 6),
                    "sampled_matched": len(deltas_matched),
                    "sampled_unmatched": len(deltas_unmatched),
                    "paired_delta_ms": {
                        "matched": _percentiles(deltas_matched),
                        "unmatched": _percentiles(deltas_unmatched),
                    },
                    "traffic_weighted_p50_delta_ms": (
                        None if pooled is None else round(pooled, 4)
                    ),
                    "within_budget": (
                        None if pooled is None else bool(pooled <= LATENCY_BUDGET_MS)
                    ),
                    "per_query": rows,
                }
            )
    finally:
        store.close()

    report = {
        "artifact": str(args.out_json),
        "what": (
            "Paired memory_recall latency at each candidate anchor match floor, on "
            "the calibration set and the calibration anchor corpus."
        ),
        "method": {
            "database": f"copy of {args.snapshot}; the live database was never opened",
            "arm_order": "alternated every iteration",
            "paired": "per-query medians differenced",
            "iters_per_query": args.iters,
            "warmup_per_query": args.warmup,
            "reads": "log_access=False, log_event=False",
        },
        "budget_ms": LATENCY_BUDGET_MS,
        "measurements": measurements,
    }
    Path(args.out_json).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.out_json}", file=sys.stderr)
    return 0


def _stride(items: Sequence[GoldsetItem], sample: int) -> list[GoldsetItem]:
    """Up to ``sample`` items, spread evenly over the ``query_id`` order."""

    ordered = sorted(items, key=lambda item: item.query_id)
    if sample <= 0 or len(ordered) <= sample:
        return list(ordered)
    step = len(ordered) / sample
    return [ordered[int(index * step)] for index in range(sample)]


def _mixture_p50(
    matched: Sequence[float], unmatched: Sequence[float], matched_share: float
) -> float | None:
    """p50 of the delta distribution real traffic would see at this floor.

    The two strata are sampled evenly, but traffic is not: the operator pays
    the matched delta on ``matched_share`` of queries and the unmatched delta
    on the rest. Reweighting the pooled sample back to that mix is what turns
    two per-stratum medians into the one number the +5 ms budget is written
    against.
    """

    if not matched and not unmatched:
        return None
    weighted: list[tuple[float, float]] = []
    if matched:
        weight = matched_share / len(matched)
        weighted.extend((value, weight) for value in matched)
    if unmatched:
        weight = (1.0 - matched_share) / len(unmatched)
        weighted.extend((value, weight) for value in unmatched)
    weighted.sort()
    total = sum(weight for _, weight in weighted)
    running = 0.0
    for value, weight in weighted:
        running += weight
        if running >= total / 2:
            return value
    return weighted[-1][0]


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------

#: Nearest in-scope anchor cosine per *scored* eval query, as published in
#: ``result.md`` §5. Quoted (not recomputed) so the calibration can show that
#: its own query population sits in the same band as the one the chosen floor
#: will meet, without opening the scored goldset to measure it.
SCORED_NEAREST_ANCHOR_COSINE = {
    "content_grounded": {"min": 0.341, "p50": 0.558, "p90": 0.710, "max": 1.000},
    "cross_lingual": {"min": 0.351, "p50": 0.455, "p90": 0.589, "max": 0.663},
    "role_query": {"min": 0.354, "p50": 0.529, "p90": 0.697, "max": 0.912},
    "source": "result.md §5, 343-item holdout against the 3,071-anchor scored corpus",
}


def _cell(report: Mapping[str, Any], threshold: float, limit: int) -> dict[str, Any]:
    for cell in report["cells"]:
        if abs(cell["threshold"] - threshold) < 1e-9 and cell["limit"] == limit:
            return cell
    raise SystemExit(f"no swept cell at threshold={threshold} limit={limit}")


def reverify_disjointness(
    calibration_set: Path, scored_paths: Sequence[str]
) -> dict[str, Any]:
    """Re-prove, at publication time, that the published set is not the answer key.

    ``build-set`` already filtered against the scored goldsets, but the artifact
    should not have to be trusted on the strength of the run that produced it:
    the file that actually ships is re-checked against the files it must be
    disjoint from. Identity only -- fingerprints, normalised queries and source
    event ids -- so no embedding model is needed and no scored text, label or
    score is read.
    """

    norm_hashes: set[str] = set()
    fingerprints: set[str] = set()
    event_ids: set[str] = set()
    scanned = 0
    for path in scored_paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            query = str(record.get("query") or "")
            if not query:
                continue
            scanned += 1
            norm_hashes.add(_norm_hash(query))
            fingerprints.add(recall_fingerprint(query, record.get("scope") or "global"))
            fingerprints.add(recall_fingerprint(query, "global"))
            if record.get("source_event_id"):
                event_ids.add(str(record["source_event_id"]))

    collisions: dict[str, list[str]] = defaultdict(list)
    items = load_goldset(calibration_set)
    for item in items:
        if _norm_hash(item.query) in norm_hashes:
            collisions["normalized_query"].append(item.query_id)
        scope = item.scope or "global"
        if (
            recall_fingerprint(item.query, scope) in fingerprints
            or recall_fingerprint(item.query, "global") in fingerprints
        ):
            collisions["recall_fingerprint"].append(item.query_id)
        source_event_id = getattr(item, "source_event_id", None)
        if source_event_id and str(source_event_id) in event_ids:
            collisions["source_event_id"].append(item.query_id)

    return {
        "what": (
            "Re-run at publication time against the shipped calibration set, so the "
            "leak claim does not rest on the build run that produced it."
        ),
        "scored_sources": list(scored_paths),
        "scored_items_scanned": scanned,
        "calibration_items": len(items),
        "collisions": {key: value for key, value in collisions.items()},
        "disjoint": not collisions,
    }


def cmd_publish(args: argparse.Namespace) -> int:
    """Assemble the measured runs into the calibration artifacts."""

    build = json.loads(Path(args.build_report).read_text(encoding="utf-8"))
    sweep = json.loads(Path(args.sweep).read_text(encoding="utf-8"))
    latency = json.loads(Path(args.latency).read_text(encoding="utf-8"))
    funnels = {
        str(path): json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.diagnose
    }
    chosen = _cell(sweep, args.threshold, args.limit)
    chosen_latency = next(
        (
            row
            for row in latency["measurements"]
            if abs(row["threshold"] - args.threshold) < 1e-9
        ),
        None,
    )
    reverified = reverify_disjointness(
        Path(build["artifact"]), build["purity"]["scored_sources"]
    )
    if not reverified["disjoint"]:
        raise SystemExit(
            "REFUSING TO PUBLISH: the calibration set collides with a scored goldset: "
            f"{reverified['collisions']}"
        )
    chosen_coverage_share = chosen["coverage"]["share"]
    shipped_share = _cell(sweep, 0.80, args.limit)["coverage"]["share"]
    coverage_multiple = (
        chosen_coverage_share / shipped_share if shipped_share else float("inf")
    )
    # The widest funnel measured -- the mechanism at its theoretical maximum, which
    # is what makes "the floor is not the constraint" a measurement and not a claim.
    widest_funnel = min(
        (data["funnel"] | {"threshold": data["threshold"]} for data in funnels.values()),
        key=lambda data: data["threshold"],
    )
    report = {
        "artifact": str(args.out_json),
        "what": (
            "Calibration of the query-anchor match entry "
            "(ANCHOR_MATCH_COSINE_THRESHOLD, ANCHOR_MATCH_LIMIT) on a calibration set "
            "built strictly inside the scored run's anchor training window."
        ),
        "decision": {
            "ANCHOR_MATCH_COSINE_THRESHOLD": args.threshold,
            "ANCHOR_MATCH_LIMIT": args.limit,
            "previous": {"threshold": 0.80, "limit": 5},
            "applied_unchanged_to_the_scored_run": True,
            "statement": (
                f"ANCHOR_MATCH_COSINE_THRESHOLD = {args.threshold} and "
                f"ANCHOR_MATCH_LIMIT = {args.limit} are written into "
                "src/living_memory/query_anchors.py and are the values the scored "
                "temporal-holdout run is to be executed with, unchanged. Nothing "
                "downstream may retune them against the scored goldset."
            ),
            "measured_at_the_chosen_value": {
                "coverage": chosen["coverage"],
                "precision": chosen["precision"],
                "quality": {
                    "overall": chosen["quality"]["overall"],
                    "per_stratum": chosen["quality"]["per_stratum"],
                    "ranking_changed_items": chosen["quality"]["ranking_changed_items"],
                    "items_improved": chosen["quality"]["items_improved"],
                    "items_regressed": chosen["quality"]["items_regressed"],
                },
                "cost": chosen["cost"],
                "paired_latency": chosen_latency,
            },
        },
        "leak_rule": {
            "reverified_at_publication": reverified,
            "scored_sets_never_read_for_selection": [
                "/tmp/anchor-eval/goldset-holdout-20260715.jsonl (343 items)",
                "artifacts/harness/goldset.jsonl (74 curated items)",
            ],
            "how_they_were_touched": (
                "Opened once by load_exclusions to derive fingerprints, source event "
                "ids, normalised-query hashes and query embeddings, used only to DROP "
                "overlapping calibration items. No scored query text, relevance label "
                "or score was read, stored or reported."
            ),
            "dropped_by_that_filter": build["purity"]["dropped"],
            "temporal": build["temporal"],
            "purity": build["purity"],
        },
        "split": {
            "anchor_window": f"events < {build['window']['start']}",
            "calibration_query_window": (
                f"{build['window']['start']} <= event < {build['window']['end']}"
            ),
            "scored_window": f"event >= {build['window']['end']}",
            "why_disjoint": (
                "Calibration queries are the events that BUILD the scored run's anchor "
                "corpus, so they are disjoint from the scored, post-cutoff items by "
                "construction. The two untimestamped strata are newly authored and "
                "their disjointness is asserted by the purity filter above."
            ),
            "anchor_corpus": sweep["anchor_corpus"],
            "snapshot_sha256": sweep["snapshot_sha256"],
        },
        "calibration_set": {
            "path": build["artifact"],
            "sha256": sweep["calibration_set_sha256"],
            "items": build["items"],
            "per_stratum": build["per_stratum"],
            "content_grounded": build["content_grounded"],
            "cross_lingual": {
                key: value
                for key, value in build["cross_lingual"].items()
                if key != "rule_audit"
            },
            "cross_lingual_rule_audit": build["cross_lingual"]["rule_audit"],
            "role_query": build["role_query"],
        },
        "external_validity": {
            "calibration_nearest_anchor_cosine": sweep["nearest_anchor_cosine"],
            "scored_nearest_anchor_cosine": SCORED_NEAREST_ANCHOR_COSINE,
            "reading": (
                "The calibration queries face a 1,607-anchor corpus and the scored "
                "queries a 3,071-anchor one, so the two nearest-anchor distributions "
                "are the thing that decides whether a floor calibrated here transfers. "
                "They sit in the same band, and they do not shift in one direction: "
                "cross_lingual is higher here (p50 0.508 vs 0.455), role_query lower "
                "(0.451 vs 0.529), content_grounded within 0.02."
            ),
        },
        "baseline_no_anchors": sweep["baseline_no_anchors"],
        "sweep": {"grid": sweep["grid"], "cells": sweep["cells"]},
        "funnel": {
            path: {"threshold": data["threshold"], "limit": data["limit"], **data["funnel"],
                   "per_stratum": data["per_stratum"]}
            for path, data in sorted(funnels.items())
        },
        "funnel_detail_at_the_widest_floor": next(
            (
                data["items_with_a_relevant_edge"]
                for data in funnels.values()
                if abs(data["threshold"] - 0.55) < 1e-9
            ),
            [],
        ),
        "latency": latency,
        "inert_path_check": sweep["inert_path_check"],
        "prediction_for_the_scored_run": {
            "claim": (
                f"At {args.threshold} the entry will fire on far more of the scored "
                "holdout than the 13 of 343 (3.8%) it fired on at 0.80 -- this "
                f"calibration measures {chosen_coverage_share:.1%} -- and will still not "
                "move hit@5 on either target stratum."
            ),
            "why": (
                f"Measured here over {len(sweep['cells'])} cells: coverage rises "
                f"{coverage_multiple:.0f}x from the shipped floor, and hit@1/hit@5/hit@10 "
                "come back identical to the anchors-off arm in EVERY cell -- including "
                "at no floor at all, where all 234 queries match. Zero items improved "
                "anywhere in the grid. The funnel names the binding leak, and it is not "
                f"the floor: at no floor only {widest_funnel['matched_with_relevant_edge']} "
                f"of {widest_funnel['matched']} matched anchors carry an edge to a "
                f"relevant node, and of the "
                f"{widest_funnel['relevant_edge_and_baseline_missed_at_5']} items whose "
                "relevant node WAS seeded and WAS missing from the anchors-off top 5, "
                f"{widest_funnel['rescued_into_top_5']} entered it."
            ),
            "falsifier": (
                f"A scored run at {args.threshold} that lifts cross_lingual hit@5 or "
                "role_query hit@1 refutes the second half; one that leaves coverage near "
                "3.8% refutes the first. Either outcome means the calibration set's "
                "anchor corpus is not representative of the scored one, which is the "
                "assumption the external-validity section exposes."
            ),
        },
        "reproduce": [
            "python3 scripts/backfill_query_anchors.py backfill --db SNAP "
            "--existing-backup BASE --until 2026-06-10T00:00:00Z",
            "python3 scripts/anchor_match_calibration.py build-set --snapshot SNAP "
            "--working-db WORK --scored-goldset HOLDOUT --scored-goldset "
            "artifacts/harness/goldset.jsonl --out SET --report BUILD",
            "python3 scripts/anchor_match_calibration.py sweep --snapshot SNAP "
            "--working-db WORK --calibration-set SET --out-json SWEEP",
            "python3 scripts/anchor_match_calibration.py diagnose --snapshot SNAP "
            "--working-db WORK --calibration-set SET --threshold 0.55 --out-json DIAG",
            "python3 scripts/anchor_match_calibration.py latency --snapshot SNAP "
            "--working-db WORK --calibration-set SET --threshold 0.55 --threshold 0.60 "
            "--threshold 0.65 --threshold 0.80 --out-json LAT",
            "python3 scripts/anchor_match_calibration.py publish --build-report BUILD "
            "--sweep SWEEP --latency LAT --diagnose DIAG ... --threshold 0.60 --limit 5 "
            "--out-json artifacts/anchors/calibration.json "
            "--out-md artifacts/anchors/calibration.md",
        ],
    }
    Path(args.out_json).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    Path(args.out_md).write_text(_render_markdown(report), encoding="utf-8")
    print(f"wrote {args.out_json} and {args.out_md}", file=sys.stderr)
    return 0


def _render_markdown(report: Mapping[str, Any]) -> str:
    decision = report["decision"]
    threshold = decision["ANCHOR_MATCH_COSINE_THRESHOLD"]
    limit = decision["ANCHOR_MATCH_LIMIT"]
    chosen = decision["measured_at_the_chosen_value"]
    lines: list[str] = []
    add = lines.append

    add("# Query-anchor match entry — calibration")
    add("")
    add(
        f"**Chosen: `ANCHOR_MATCH_COSINE_THRESHOLD = {threshold:.2f}`, "
        f"`ANCHOR_MATCH_LIMIT = {limit}`**, written into "
        "`src/living_memory/query_anchors.py`. "
        "These are the values the scored temporal-holdout run is executed with, "
        "**unchanged** — nothing downstream may retune them against the scored goldset."
    )
    add("")
    add(report["what"])
    add("")
    add("---")
    add("")
    add("## 1. The split, and why it is not the answer key")
    add("")
    split = report["split"]
    add("```")
    add("|------- calibration anchors -------|-- calibration queries --|--- SCORED ---|")
    add("<                    2026-06-10                    2026-07-15               >")
    add("```")
    add("")
    add(
        f"Anchors: `{split['anchor_window']}` — "
        f"**{split['anchor_corpus']['live_anchors']} live anchors / "
        f"{split['anchor_corpus']['edges']} edges**, built by "
        "`backfill_query_anchors.py --until 2026-06-10T00:00:00Z` into a copy of the "
        f"anchor-free base snapshot `{split['snapshot_sha256'][:16]}…`. "
        "The live database was never opened."
    )
    add("")
    add(split["why_disjoint"])
    add("")
    leak = report["leak_rule"]
    add("**The scored sets were opened once, to exclude.** " + leak["how_they_were_touched"])
    add("")
    purity = leak["purity"]
    add(
        f"| scored items scanned | calibration candidates | dropped | kept |\n"
        f"|---|---|---|---|\n"
        f"| {purity['scored_items_scanned']} | {purity['examined']} | "
        f"**{purity['dropped']}** | {purity['kept']} |"
    )
    add("")
    reasons: dict[str, int] = defaultdict(int)
    for entry in purity["dropped_detail"]:
        reasons[entry["reason"]] += 1
    add(
        "Drop reasons: "
        + ", ".join(f"`{reason}` {count}" for reason, count in sorted(reasons.items()))
        + ". That 24 of 258 candidates collided is the point of running the filter: the "
        "shipped goldset's `content_grounded` stratum was itself built at cutoff "
        "2026-06-10, so it overlaps this window and would have leaked."
    )
    add("")
    kept = purity["nearest_scored_cosine_kept"]["all"]
    add(
        f"After the filter, the closest any calibration query sits to any scored query "
        f"is **{kept['max']}** cosine (p50 {kept['p50']}), under the "
        f"{purity['near_dup_cosine']} near-duplicate cut."
    )
    add("")
    reverified = leak.get("reverified_at_publication")
    if reverified:
        add(
            f"**Re-verified at publication**, against the file that actually ships: "
            f"{reverified['calibration_items']} calibration items checked against "
            f"{reverified['scored_items_scanned']} scored items on recall fingerprint, "
            f"normalised query and source event id — "
            f"**{sum(len(v) for v in reverified['collisions'].values())} collisions**. "
            "`publish` refuses to write the artifact if that number is not zero, so the "
            "leak claim is enforced rather than reported."
        )
        add("")
    temporal = leak["temporal"]
    add(
        f"Temporal assertion: **{temporal['timestamped_items']}** timestamped items, "
        f"earliest `{temporal['earliest']}`, latest `{temporal['latest']}`, "
        f"**{temporal['outside_window']} outside the window**. The other "
        f"{temporal['untimestamped_items']} carry no timestamp by construction and are "
        "bound by the fingerprint/near-duplicate test instead."
    )
    add("")
    add("## 2. The calibration set")
    add("")
    calset = report["calibration_set"]
    add(
        f"**{calset['items']} items** — "
        + ", ".join(f"`{name}` {count}" for name, count in calset["per_stratum"].items())
        + f" — sha256 `{calset['sha256'][:16]}…`."
    )
    add("")
    add(
        f"`content_grounded` is the shipped `build_content_grounded` label (IDF "
        f"containment against the consuming trace) over "
        f"{calset['content_grounded']['inside_window']} eligible in-window events, "
        f"capped at {calset['content_grounded']['cap']} and sampled with "
        f"`{calset['content_grounded']['sampling']}`."
    )
    add("")
    add(
        f"`cross_lingual` and `role_query` are **newly authored** — "
        f"`artifacts/harness/seed-queries.json` holds exactly the 38 + 36 items already "
        f"inside the scored goldset, so there was no surplus to borrow. "
        f"{calset['cross_lingual']['authored_specs']} jargon/paraphrase pairs were "
        f"authored under the recorded `cross_lingual_rule` and re-verified mechanically "
        f"on this snapshot (paraphrase top-1 vector ≥ 0.40, node predominantly Latin, "
        f"jargon strictly lower on that node and not already ranking it first): "
        f"**{calset['cross_lingual']['kept_after_rule_recheck']} kept**. "
        f"{calset['role_query']['authored_specs']} role/procedural queries were authored "
        f"against active `level:schema` nodes carrying `context.trigger`, re-validated by "
        f"the shipped `build_role_query`."
    )
    add("")
    add("## 3. Does a floor calibrated here transfer?")
    add("")
    add(
        "The one distribution that decides it is the nearest in-scope anchor cosine per "
        "query — the quantity the floor is compared against. Calibration (1,607 anchors) "
        "against scored (3,071 anchors, `result.md` §5):"
    )
    add("")
    add("| stratum | calibration p50 / p90 / max | scored p50 / p90 / max |")
    add("|---|---|---|")
    scored = report["external_validity"]["scored_nearest_anchor_cosine"]
    for stratum in ("content_grounded", "cross_lingual", "role_query"):
        mine = report["external_validity"]["calibration_nearest_anchor_cosine"][stratum]
        theirs = scored[stratum]
        add(
            f"| `{stratum}` | {mine['p50']:.3f} / {mine['p90']:.3f} / {mine['max']:.3f} "
            f"| {theirs['p50']:.3f} / {theirs['p90']:.3f} / {theirs['max']:.3f} |"
        )
    add("")
    add(report["external_validity"]["reading"])
    add("")
    add("## 4. The sweep")
    add("")
    add(
        "Coverage, precision, quality and cost at every swept floor, at "
        "`ANCHOR_MATCH_LIMIT = 5`. `anchor-prec` is the share of *matched anchors* "
        "holding an edge to a relevant node; `item-prec` is the share of *matched "
        "queries* with at least one such anchor."
    )
    add("")
    add(
        "| floor | coverage | cross_lingual | role_query | anchor-prec | item-prec "
        "| hit@5 | MRR | rank changes | measured p50 Δ (traffic-weighted) |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|")
    latency_by_threshold = {
        row["threshold"]: row for row in report["latency"]["measurements"]
    }
    for cell in report["sweep"]["cells"]:
        if cell["limit"] != 5:
            continue
        coverage = cell["coverage"]
        cross = coverage["per_stratum"].get("cross_lingual", {})
        role = coverage["per_stratum"].get("role_query", {})
        row = latency_by_threshold.get(cell["threshold"])
        # Where two replicates exist, show the range rather than one run: at 0.55
        # they straddle the budget, and a single number would hide that.
        runs = (
            (report["latency"].get("replicates") or {})
            .get("per_threshold", {})
            .get(f"{cell['threshold']}", {})
        )
        deltas = [
            run["traffic_weighted_p50_delta_ms"]
            for run in runs.values()
            if run.get("traffic_weighted_p50_delta_ms") is not None
        ]
        if len(deltas) > 1 and abs(max(deltas) - min(deltas)) > 1e-9:
            cost = f"**{min(deltas):+.2f} … {max(deltas):+.2f} ms**"
        elif row and row["traffic_weighted_p50_delta_ms"] is not None:
            cost = f"**{row['traffic_weighted_p50_delta_ms']:+.2f} ms**"
        else:
            cost = "—"
        if deltas and max(deltas) > LATENCY_BUDGET_MS:
            cost += " ⚠ over"
        elif row and row.get("within_budget") is False:
            cost += " ⚠ over"
        mark = " **←**" if abs(cell["threshold"] - threshold) < 1e-9 else ""
        add(
            f"| {cell['threshold']:.2f}{mark} | {coverage['matched_items']}/{coverage['items']} "
            f"({coverage['share']:.1%}) | {cross.get('matched', 0)}/{cross.get('items', 0)} "
            f"| {role.get('matched', 0)}/{role.get('items', 0)} "
            f"| {cell['precision']['anchor_precision']} | {cell['precision']['item_precision']} "
            f"| {cell['quality']['overall']['hit@5']:.4f} | {cell['quality']['overall']['mrr']:.4f} "
            f"| {cell['quality']['ranking_changed_items']} "
            f"(+{cell['quality']['items_improved']}/−{cell['quality']['items_regressed']}) | {cost} |"
        )
    add("")
    baseline = report["baseline_no_anchors"]["overall"]
    add(
        f"Anchors-off baseline on the same set: hit@1 {baseline['hit@1']:.4f}, "
        f"hit@5 **{baseline['hit@5']:.4f}**, hit@10 {baseline['hit@10']:.4f}, "
        f"MRR **{baseline['mrr']:.4f}**."
    )
    add("")
    add("### The limit")
    add("")
    add(
        "`ANCHOR_MATCH_LIMIT` does not move coverage at all — coverage asks whether "
        "*any* anchor clears the floor, which no limit changes. It moves precision and "
        "risk:"
    )
    add("")
    add("| floor | L=3 item-prec | L=5 item-prec | L=10 item-prec | L=10 regressions |")
    add("|---|---|---|---|---|")
    for floor in (0.55, 0.60, 0.65):
        cells = {
            cell["limit"]: cell
            for cell in report["sweep"]["cells"]
            if abs(cell["threshold"] - floor) < 1e-9
        }
        add(
            f"| {floor:.2f} | {cells[3]['precision']['item_precision']} "
            f"| {cells[5]['precision']['item_precision']} "
            f"| {cells[10]['precision']['item_precision']} "
            f"| {cells[10]['quality']['items_regressed']} |"
        )
    add("")
    l5_regressions = sum(
        cell["quality"]["items_regressed"]
        for cell in report["sweep"]["cells"]
        if cell["limit"] == 5 and cell["threshold"] >= 0.55
    )
    l10_regressions = sum(
        cell["quality"]["items_regressed"]
        for cell in report["sweep"]["cells"]
        if cell["limit"] == 10 and cell["threshold"] >= 0.55
    )
    add(
        f"`L = 5` dominates `L = 3` on item precision at every floor, and over the "
        f"in-budget part of the grid (floors ≥ 0.55) it regressed **{l5_regressions}** "
        f"items against `L = 10`'s **{l10_regressions}**. It stays at **5**. Below 0.55 "
        "every limit regresses items, which is a second reason not to go there."
    )
    add("")
    add("## 5. Cost")
    add("")
    method = report["latency"]["method"]
    sampled = max(
        (row["sampled_matched"] for row in report["latency"]["measurements"]), default=0
    )
    add(
        "`artifacts/anchors/latency.json` measured the matched stratum on **two** "
        "queries. A floor is chosen partly on what it costs, so it was re-measured here "
        "with the same paired design (arms alternated every iteration, per-query medians "
        f"differenced, {method['iters_per_query']} iterations after "
        f"{method['warmup_per_query']} warmups, reads that log neither access nor event) "
        f"on up to {sampled} matched and {sampled} unmatched calibration queries per "
        "floor."
    )
    add("")
    add("| floor | matched share | matched paired p50 | unmatched paired p50 | traffic-weighted p50 | +5 ms budget |")
    add("|---|---|---|---|---|---|")
    for row in report["latency"]["measurements"]:
        matched = row["paired_delta_ms"]["matched"]
        unmatched = row["paired_delta_ms"]["unmatched"]
        verdict = "**over**" if row["within_budget"] is False else "inside"
        # At no floor every query matches, so the unmatched arm is empty.
        cell_matched = f"{matched['p50']:+.2f} ms" if matched else "—"
        cell_unmatched = f"{unmatched['p50']:+.2f} ms" if unmatched else "— (none)"
        add(
            f"| {row['threshold']:.2f} | {row['matched_share']:.1%} "
            f"| {cell_matched} | {cell_unmatched} "
            f"| **{row['traffic_weighted_p50_delta_ms']:+.2f} ms** | {verdict} |"
        )
    add("")
    add(
        "The cost is not the anchor scan; it is the **second graph walk** a matched "
        "anchor opens. It therefore scales with the walk it duplicates: on small-scope "
        "queries (~90 ms baseline) a match costs +5 to +10 ms, and on the deep "
        "large-scope queries (200–1000 ms baseline) it costs +100 to +500 ms. That is "
        "why the budget is crossed by *match rate*, not by threshold as such."
    )
    replicates = report["latency"].get("replicates")
    if replicates:
        add("")
        add("### The cost measurement is noisy, so it was run twice")
        add("")
        add(replicates["what"])
        add("")
        add(
            "| floor | replicate 1 (n=20) | replicate 2 (n=25) | both inside +5 ms? |"
        )
        add("|---|---|---|---|")
        for key, runs in replicates["per_threshold"].items():
            first = runs.get("r1_sample20")
            second = runs.get("r2_sample25")
            fmt = (
                lambda run: f"{run['traffic_weighted_p50_delta_ms']:+.2f} ms"
                if run
                else "—"
            )
            inside = all(run["within_budget"] for run in runs.values())
            verdict = "**yes**" if inside else "**no** — straddles or over"
            mark = " **←**" if abs(float(key) - threshold) < 1e-9 else ""
            add(f"| {float(key):.2f}{mark} | {fmt(first)} | {fmt(second)} | {verdict} |")
        add("")
        add(
            "The two runs disagree at 0.55 by nearly 2× and land on opposite sides of "
            "the budget. That is not a defect in either run — it is what a heavy-tailed "
            "delta does to a p50 at n≈20 — and it is the reason 0.55 is rejected as "
            "*uncertifiable* rather than as *measured over*. 0.60 came back inside the "
            "budget on both."
        )
    add("")
    add("## 6. Why quality is flat, measured rather than asserted")
    add("")
    add(
        "Coverage moves by two orders of magnitude across the grid and the number of "
        "calibration items whose ranking changes stays at 0. The funnel separates the "
        "two possible causes — the edges point nowhere useful, or the ranker will not "
        "surface what they point at:"
    )
    add("")
    add("| floor | matched | …with an edge to a relevant node | …that the baseline missed at 5 | …rescued into top 5 |")
    add("|---|---|---|---|---|")
    for _, data in sorted(
        report["funnel"].items(), key=lambda entry: -entry[1]["threshold"]
    ):
        add(
            f"| {data['threshold']:.2f} | {data['matched']} "
            f"| {data['matched_with_relevant_edge']} "
            f"| {data['relevant_edge_and_baseline_missed_at_5']} "
            f"| **{data['rescued_into_top_5']}** |"
        )
    add("")
    widest = min(report["funnel"].values(), key=lambda data: data["threshold"])
    add(
        "**Both leaks are real, and the first is the larger.** Read the bottom row, "
        f"which is the mechanism at its theoretical maximum: with **no floor at all** "
        f"every one of the {widest['matched']} queries matches an anchor, and still only "
        f"**{widest['matched_with_relevant_edge']}** of them match an anchor whose edges "
        "lead to anything the item marks relevant. Of those, most were already answered "
        f"at rank ≤ 5 without anchors, leaving "
        f"**{widest['relevant_edge_and_baseline_missed_at_5']}** items the entry could "
        f"possibly have rescued — and it rescued **{widest['rescued_into_top_5']}**. "
        f"`role_query` is the sharpest case: all "
        f"{widest['per_stratum']['role_query']['matched']} of its queries match at no "
        "floor and **not one** matched anchor carries an edge to a relevant node."
    )
    add("")
    add(
        "Where the seeded node did not surface, the funnel records why: it did not enter "
        "the returned list at all (`seeded_relevant_node_rank: null`), at an activation "
        "of 0.14–0.21 — a cosine times an edge weight — against a learned graph channel "
        "weight that cannot lift it past a bm25/vector-dominated ranking."
    )
    add("")
    add(
        "So the floor is **not** the last thing standing between this mechanism and its "
        "targets. Lowering it is still right — it is measurably mis-set, and it is the "
        "only thing this node owns — but the next lever is the edge yield per anchor and "
        "the activation an anchor seed carries into the ranker, neither of which is in "
        "scope here."
    )
    add("")
    add("## 7. The choice")
    add("")
    add(
        f"**{threshold:.2f} at limit {limit}.** The reasoning, in the order the evidence "
        "forces:"
    )
    add("")
    shipped_cell = _cell(report["sweep"], 0.80, 5)
    shipped_matched = shipped_cell["coverage"]["matched_items"]
    shipped_items = shipped_cell["coverage"]["items"]
    floors_swept = sorted({cell["threshold"] for cell in report["sweep"]["cells"]})
    improved_anywhere = sum(
        cell["quality"]["items_improved"] for cell in report["sweep"]["cells"]
    )
    add(
        f"1. **0.80 is measurably wrong.** It admits {shipped_matched} of "
        f"{shipped_items} calibration queries "
        f"({shipped_cell['coverage']['share']:.1%}), which reproduces the 13 of 343 "
        "(3.8%) the scored run measured. A floor that never fires cannot be evaluated, "
        "and was never calibrated — it was picked to sit under the 0.95 dedup constant."
    )
    add(
        f"2. **Quality cannot choose.** All {len(report['sweep']['cells'])} cells — "
        f"every floor from **no floor at all** ({floors_swept[0]:.2f}, where all 234 "
        f"queries match) up to {floors_swept[-1]:.2f}, at limits 3, 5 and 10 — return "
        f"the anchors-off hit@1/hit@5/hit@10 exactly, with "
        f"**{improved_anywhere} items improved anywhere in the grid**. §6 says why. So "
        "the choice is coverage against cost, not quality against cost."
    )
    add(
        "3. **A wrong match is cheap in quality and expensive in time.** Since the "
        "graph-floor monotonicity fix an anchor activation cannot lower a candidate's "
        "score, so the cost of admitting a bad anchor is the walk it opens — which is "
        "exactly what the +5 ms p50 budget already prices. That makes the budget, not "
        "precision, the binding constraint."
    )
    add(
        "4. **0.60 is the widest floor certified inside the budget.** Traffic-weighted "
        "paired p50, two independent replicates: 0.55 gave **+8.31 ms and +4.17 ms** — "
        "landing on opposite sides of the hard +5 ms budget, so it cannot be certified "
        "at all — while 0.60 gave **+2.07 ms and +2.51 ms**, inside on both. No floor "
        "at all costs **+25.93 ms**. Width is the whole point, so take the widest floor "
        "the budget actually certifies."
    )
    add(
        f"5. **It buys real width on the class in question.** `cross_lingual` coverage "
        f"goes from 1 of 34 at 0.80 to **7 of 34** at 0.60. `role_query` stays at 0 of 44 "
        "here — its calibration items sit at nearest-anchor p50 0.451 / max 0.593 against "
        "the smaller calibration corpus — but the scored `role_query` population sits at "
        "p50 0.529 / p90 0.697, so 0.60 reaches roughly its top third rather than the "
        "1 of 36 that 0.80 reached."
    )
    add("")
    add(
        "Rejected: **0.55**, which buys the most coverage (53.4%, and the only "
        "`role_query` matches on this set) but whose cost straddles the budget across "
        "replicates; **0.65**, which costs almost nothing (+0.12 / +0.37 ms) but gives "
        "up more than half of 0.60's coverage (12.4% vs 29.5%) to buy precision that — "
        "per §6 — no longer converts into quality; **0.70+**, which is 0.80's failure "
        "mode with a smaller number on it; and **no floor at all**, which is the "
        "cleanest disproof of the premise: it fires on 100% of queries, improves "
        "nothing, and costs +25.93 ms."
    )
    add("")
    prediction = report["prediction_for_the_scored_run"]
    add("## 8. What this predicts, and how to falsify it")
    add("")
    add(f"**{prediction['claim']}**")
    add("")
    add(prediction["why"])
    add("")
    add(f"*Falsifier.* {prediction['falsifier']}")
    add("")
    add("---")
    add("")
    inert = report["inert_path_check"]
    add(
        f"Method note: the sweep reuses an item's anchors-off ranking for every cell in "
        f"which that item matches nothing. That reuse is checked rather than assumed — "
        f"{inert['checked']} items were re-run for real with anchors ON at an "
        f"unreachable floor and {inert['identical']} came back identical "
        f"(`inert_path_check.ok = {inert['ok']}`)."
    )
    add("")
    add("Full record: `artifacts/anchors/calibration.json`.")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anchor_match_calibration.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build-set", help="Build the calibration set from the calibration snapshot."
    )
    build.add_argument("--snapshot", required=True, type=Path)
    build.add_argument("--working-db", required=True, type=Path)
    build.add_argument("--refresh-working", action="store_true")
    build.add_argument("--window-start", default=DEFAULT_WINDOW_START)
    build.add_argument("--window-end", default=DEFAULT_WINDOW_END)
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument(
        "--max-content-grounded", type=int, default=DEFAULT_CONTENT_GROUNDED_CAP
    )
    build.add_argument(
        "--scored-goldset",
        action="append",
        required=True,
        help="A scored goldset, opened ONLY to exclude overlapping items. Repeatable.",
    )
    build.add_argument("--near-dup-cosine", type=float, default=DEFAULT_NEAR_DUP_COSINE)
    build.add_argument("--out", required=True, type=Path)
    build.add_argument("--report", required=True, type=Path)
    build.set_defaults(func=cmd_build_set)

    sweep = subparsers.add_parser("sweep", help="Sweep the match floor and limit.")
    sweep.add_argument("--snapshot", required=True, type=Path)
    sweep.add_argument("--working-db", required=True, type=Path)
    sweep.add_argument("--refresh-working", action="store_true")
    sweep.add_argument("--calibration-set", required=True, type=Path)
    sweep.add_argument("--threshold", action="append", type=float, default=[])
    sweep.add_argument("--limit", action="append", type=int, default=[])
    sweep.add_argument("--verify-inert", type=int, default=12)
    sweep.add_argument("--out-json", required=True, type=Path)
    sweep.set_defaults(func=cmd_sweep)

    diagnose = subparsers.add_parser(
        "diagnose", help="Per-item funnel from a matched anchor to a rank change."
    )
    diagnose.add_argument("--snapshot", required=True, type=Path)
    diagnose.add_argument("--working-db", required=True, type=Path)
    diagnose.add_argument("--refresh-working", action="store_true")
    diagnose.add_argument("--calibration-set", required=True, type=Path)
    diagnose.add_argument("--threshold", type=float, required=True)
    diagnose.add_argument("--limit", type=int, default=5)
    diagnose.add_argument("--out-json", required=True, type=Path)
    diagnose.set_defaults(func=cmd_diagnose)

    latency = subparsers.add_parser(
        "latency", help="Paired recall latency at each candidate match floor."
    )
    latency.add_argument("--snapshot", required=True, type=Path)
    latency.add_argument("--working-db", required=True, type=Path)
    latency.add_argument("--refresh-working", action="store_true")
    latency.add_argument("--calibration-set", required=True, type=Path)
    latency.add_argument("--threshold", action="append", type=float, required=True)
    latency.add_argument("--limit", type=int, default=5)
    latency.add_argument("--sample", type=int, default=20)
    latency.add_argument("--iters", type=int, default=9)
    latency.add_argument("--warmup", type=int, default=2)
    latency.add_argument("--out-json", required=True, type=Path)
    latency.set_defaults(func=cmd_latency)

    publish = subparsers.add_parser(
        "publish", help="Assemble the measured runs into the calibration artifacts."
    )
    publish.add_argument("--build-report", required=True, type=Path)
    publish.add_argument("--sweep", required=True, type=Path)
    publish.add_argument("--latency", required=True, type=Path)
    publish.add_argument(
        "--diagnose",
        action="append",
        default=[],
        type=Path,
        help="A diagnose report to fold in. Repeatable.",
    )
    publish.add_argument(
        "--threshold",
        required=True,
        type=float,
        help="The chosen floor. Must name a cell the sweep actually measured.",
    )
    publish.add_argument("--limit", required=True, type=int, help="The chosen limit.")
    publish.add_argument("--out-json", required=True, type=Path)
    publish.add_argument("--out-md", required=True, type=Path)
    publish.set_defaults(func=cmd_publish)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
