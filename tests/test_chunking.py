from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from living_memory.chunking import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    ENCODER_SPECIAL_TOKENS,
    MODEL_MAX_SEQ_LENGTH,
    TOKENIZER_ENV_VAR,
    HeuristicTokenizer,
    ModelTokenizer,
    TextChunk,
    Tokenizer,
    chunk_text,
    count_tokens,
    get_tokenizer,
    reset_tokenizer_cache,
)


# A node-shaped text: Russian prose with English identifiers, numbers, bullet
# lists and paragraph breaks, the way real Living Memory content looks.
REAL_SHAPED_NODE = """Замер векторного канала recall на живой базе 2026-08-17 (12839 активных узлов, 16583 эмбеддинга): st.max_seq_length = 128, и 67% активных узлов длиннее этого лимита при медиане 235 токенов. Хвост за токеном 128 в вектор не попадает вообще — cos(вектор полного узла, вектор его первых 128 токенов) = 1.000. На узлах длиннее 800 символов модель видит 10-36% контента, в среднем около 20%.

Дешёвый вариант «поднять max_seq_length до 512» проверен и отвергнут. На восьми узлах длиннее 1500 символов с запросом из хвостового фрагмента (токены 200-320) средний cos составил 0.479 против 0.489 при лимите 128: добавление текста в один вектор разбавляет сигнал, а не обогащает его. Плюс энкод дороже примерно в пять раз — 18 текстов в секунду против 90 на CPU.

Чанкинг окнами около 128 токенов с overlap 32 и max-pool по чанкам узла даёт на том же тесте средний cos 0.954 с выигрышем на всех восьми узлах. Оговорка: запрос в этом микротесте — дословный фрагмент, на перефразировках выигрыш будет скромнее, поэтому финальный гейт — офлайн-харнесс на реальных запросах из recall_events, а не этот микротест.

Инвентарь того, что меняется в хранении:
- nodes.embedding сейчас лежит JSON-текстом: 7.9 КБ на 384-мерный вектор против 1.5 КБ в float32, то есть 135 МБ на базу.
- Холодный старт первого recall парсит 16.5k JSON примерно 750 мс, и это видно в p50 первого запроса сессии.
- float32 BLOB little-endian: около 45k чанков (в 3.53 раза больше строк, чем узлов) займут примерно 69 МБ, чтение через np.frombuffer 35-140 мс.
- Скан 45k на 384 float32 занимает около 1.2 мс, max-pool в узлы около 0.5 мс на numpy против 0.3 мс сейчас — приемлемо.
- Полный пересчёт эмбеддингов с чанкингом занимает примерно 8.5 минут CPU при batch 32.

Ограничения, зафиксированные до начала работ: модель эмбеддинга paraphrase-multilingual-MiniLM-L12-v2 не менять, max_seq_length глобально не поднимать, bm25/FTS и словарь _SYNONYMS не трогать, внешние векторные индексы вроде FAISS или sqlite-vec не вводить — чистый numpy плюс sqlite. Все эксперименты только на снапшотах: mode=ro или копия через sqlite backup API, потому что копия файла без -wal неполна.

The harness itself replays recorded recall_events end to end: a query goes through the live MemoryRecallService of the current revision, and hit@5 with MRR are computed against the labelled goldset. The replay module cannot do this on its own — it re-ranks recorded per-result scores, so it structurally cannot evaluate a change that alters the scores themselves, which is exactly what max-pool over chunk vectors does.

Голдсет собирается из трёх страт, чтобы гейт не свёлся к одному классу запросов. Первая страта — content-grounded события из recall_events, где полезность выдачи подтверждена через feedback_trace_id. Вторая — кросс-языковые пары: жаргонные русские запросы к английским узлам, класс «поревьювь / мердж-реквест / апрув», на котором замерены провалы cos 0.201-0.392 против 0.75-0.83 на нормальной лексике. Третья — ролевые запросы-промахи вида «EZ-13871 ревью задачи», где ожидается процедурная схема, а не отдельный trace.

Гейт фазы 1 сформулирован числами, а не впечатлениями: на голдсете hit@5 и MRR не ниже baseline, на хвостовом подмножестве (запрос по содержимому за пределами первых 128 токенов узла) — выраженный рост, размер хранимых эмбеддингов меньше нынешних 135 МБ, холодная инициализация векторного канала быстрее 200 мс против нынешних 750 мс, p50 memory_recall не хуже baseline плюс 5 мс. Отдельно требуется регрессионный тест по домашнему правилу «на старом коде падает»: узел с релевантным хвостом за токеном 128 и запрос по этому хвосту — на старом коде узел вне top-5, на новом в top-5.
"""


@pytest.fixture()
def heuristic() -> HeuristicTokenizer:
    return HeuristicTokenizer()


@pytest.fixture(scope="module")
def model_tokenizer() -> Tokenizer:
    """The encoder's own tokenizer, skipping when it is not cached locally."""

    tokenizer = get_tokenizer(backend="model")
    if not isinstance(tokenizer, ModelTokenizer):
        pytest.skip("model tokenizer is not available locally")
    return tokenizer


def _assert_sane(
    chunks: list[TextChunk],
    text: str,
    window: int,
    tokenizer: Tokenizer | None = None,
) -> None:
    """Assert every invariant a chunk list must satisfy for any input."""

    total = count_tokens(text, tokenizer=tokenizer or HeuristicTokenizer())
    assert chunks, "a text with tokens must produce at least one chunk"
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].token_start == 0
    assert chunks[0].char_start == 0
    assert chunks[-1].token_end == total
    assert chunks[-1].char_end == len(text)

    covered: set[int] = set()
    for previous, chunk in zip([None, *chunks[:-1]], chunks, strict=True):
        assert chunk.text, "chunks must be non-empty"
        assert chunk.text == text[chunk.char_start : chunk.char_end]
        assert 0 < chunk.token_count <= window
        assert chunk.token_start < chunk.token_end
        if previous is not None:
            assert previous.token_start <= chunk.token_start
            assert previous.token_end <= chunk.token_end
            assert chunk.token_start <= previous.token_end, "coverage must not gap"
        covered.update(range(chunk.token_start, chunk.token_end))
    assert covered == set(range(total)), "chunks must cover the whole token stream"


def test_short_text_is_one_chunk_carrying_the_whole_input(
    heuristic: HeuristicTokenizer,
) -> None:
    text = "Короткая заметка про recall и его каналы.\n\n"

    chunks = chunk_text(text, tokenizer=heuristic)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].token_start == 0
    assert chunks[0].token_end == count_tokens(text, tokenizer=heuristic)
    assert (chunks[0].char_start, chunks[0].char_end) == (0, len(text))


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t\n"])
def test_text_without_tokens_yields_no_chunks(
    text: str, heuristic: HeuristicTokenizer
) -> None:
    assert chunk_text(text, tokenizer=heuristic) == []


def test_cut_prefers_a_paragraph_break(heuristic: HeuristicTokenizer) -> None:
    first = "Alpha paragraph keeps talking about retrieval channels and weights here."
    second = "Beta paragraph starts a different topic about storage layout entirely."
    text = f"{first}\n\n{second}"
    boundary = count_tokens(first, tokenizer=heuristic)

    chunks = chunk_text(
        text, max_tokens=24, overlap_tokens=6, special_tokens=0, tokenizer=heuristic
    )

    assert len(chunks) > 1
    assert chunks[0].token_end == boundary
    assert chunks[0].text == first
    assert "Beta" not in chunks[0].text
    _assert_sane(chunks, text, window=24)


def test_cut_falls_back_to_a_sentence_break(heuristic: HeuristicTokenizer) -> None:
    sentences = [
        "Vector scores come from one embedding per node today.",
        "Chunking replaces that with several windows per node.",
        "Max pooling then folds chunk scores back into a node score.",
        "Weights were calibrated for the older distribution of scores.",
    ]
    text = " ".join(sentences)

    chunks = chunk_text(
        text, max_tokens=28, overlap_tokens=6, special_tokens=0, tokenizer=heuristic
    )

    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert chunk.text.endswith("."), chunk.text
    _assert_sane(chunks, text, window=28)


def test_russian_sentence_punctuation_is_respected(
    heuristic: HeuristicTokenizer,
) -> None:
    sentences = [
        "Первое предложение описывает векторный канал recall и его лимит.",
        "Второе предложение объясняет, почему хвост узла не виден модели!",
        "Третье предложение спрашивает, что делать с перекалибровкой весов?",
        "Четвёртое предложение фиксирует замер на живой базе от 2026-08-17.",
    ]
    text = " ".join(sentences)

    chunks = chunk_text(
        text, max_tokens=40, overlap_tokens=10, special_tokens=0, tokenizer=heuristic
    )

    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert chunk.text[-1] in ".!?", chunk.text
    assert "хвост" in " ".join(chunk.text for chunk in chunks)
    _assert_sane(chunks, text, window=40)


def test_oversized_single_sentence_is_hard_cut(heuristic: HeuristicTokenizer) -> None:
    text = " ".join(f"word{index}" for index in range(200)) + "."

    chunks = chunk_text(
        text, max_tokens=32, overlap_tokens=8, special_tokens=0, tokenizer=heuristic
    )

    assert len(chunks) > 4
    # No paragraph or sentence boundary exists inside, so the cuts are hard ones
    # that still fill the window rather than collapsing to tiny chunks.
    for chunk in chunks[:-1]:
        assert chunk.token_count >= 16
    _assert_sane(chunks, text, window=32)


def test_overlap_is_exact_when_no_boundary_exists(
    heuristic: HeuristicTokenizer,
) -> None:
    # One unbroken run of Cyrillic characters: no paragraph, sentence, or word
    # boundary anywhere, so every cut is a hard cut at the window edge.
    text = "щ" * 900

    chunks = chunk_text(
        text, max_tokens=32, overlap_tokens=8, special_tokens=0, tokenizer=heuristic
    )

    assert len(chunks) > 4
    overlaps = {
        chunks[index].token_end - chunks[index + 1].token_start
        for index in range(len(chunks) - 1)
    }
    assert overlaps == {8}
    assert {chunk.token_count for chunk in chunks[:-1]} == {32}
    _assert_sane(chunks, text, window=32)


@pytest.mark.parametrize("overlap_tokens", [0, 1, 8, 32])
def test_overlap_stays_within_the_requested_size(
    overlap_tokens: int, heuristic: HeuristicTokenizer
) -> None:
    chunks = chunk_text(
        REAL_SHAPED_NODE,
        overlap_tokens=overlap_tokens,
        tokenizer=heuristic,
    )

    assert len(chunks) > 1
    for index in range(len(chunks) - 1):
        overlap = chunks[index].token_end - chunks[index + 1].token_start
        assert 0 <= overlap <= overlap_tokens
        if overlap_tokens:
            # Snapping to a boundary may shrink the overlap, but never below half
            # of what was asked for.
            assert overlap >= max(1, overlap_tokens // 2)
    _assert_sane(
        chunks, REAL_SHAPED_NODE, window=DEFAULT_MAX_TOKENS - ENCODER_SPECIAL_TOKENS
    )


@pytest.mark.parametrize(
    "text",
    [
        "Single short sentence.",
        "Одно короткое предложение без переносов",
        REAL_SHAPED_NODE,
        REAL_SHAPED_NODE.replace("\n\n", "\n"),
        "\n".join(f"- пункт {index} списка про retrieval" for index in range(60)),
        "щ" * 500,
        "  leading and trailing whitespace \n\n\n",
        "Mixed текст with английскими словами и punctuation!!! Ещё раз?.. Да.",
    ],
)
def test_chunks_cover_the_whole_token_stream(
    text: str, heuristic: HeuristicTokenizer
) -> None:
    chunks = chunk_text(
        text, max_tokens=34, overlap_tokens=8, special_tokens=2, tokenizer=heuristic
    )

    _assert_sane(chunks, text, window=32)


def test_chunking_is_deterministic(heuristic: HeuristicTokenizer) -> None:
    runs = [
        chunk_text(REAL_SHAPED_NODE, tokenizer=HeuristicTokenizer()) for _ in range(3)
    ]

    assert runs[0] == runs[1] == runs[2]
    assert runs[0] == chunk_text(REAL_SHAPED_NODE, tokenizer=heuristic)
    assert len({tuple(chunk.text for chunk in run) for run in runs}) == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": -5}, "max_tokens"),
        ({"max_tokens": 2, "special_tokens": 2}, "content window"),
        ({"special_tokens": -1}, "special_tokens"),
        ({"overlap_tokens": -1}, "overlap_tokens"),
        # overlap at or above the content window is the zero-advance hazard.
        ({"max_tokens": 32, "overlap_tokens": 30, "special_tokens": 2}, "advance"),
        ({"max_tokens": 32, "overlap_tokens": 99, "special_tokens": 0}, "advance"),
        ({"max_tokens": 128, "overlap_tokens": 126}, "advance"),
    ],
)
def test_invalid_arguments_are_rejected(
    kwargs: dict[str, int], message: str, heuristic: HeuristicTokenizer
) -> None:
    with pytest.raises(ValueError, match=message):
        chunk_text(REAL_SHAPED_NODE, tokenizer=heuristic, **kwargs)


def test_long_real_shaped_node_yields_a_plausible_chunk_count() -> None:
    assert len(REAL_SHAPED_NODE) >= 3000

    chunks = chunk_text(REAL_SHAPED_NODE)

    window = DEFAULT_MAX_TOKENS - ENCODER_SPECIAL_TOKENS
    total = count_tokens(REAL_SHAPED_NODE)
    ideal = math.ceil((total - DEFAULT_OVERLAP_TOKENS) / (window - DEFAULT_OVERLAP_TOKENS))
    assert total > window, "the fixture must not fit in a single window"
    assert 3 <= math.ceil(total / window) <= len(chunks) <= 2 * ideal
    for chunk in chunks:
        assert chunk.token_count <= window


def test_module_imports_and_chunks_without_torch() -> None:
    """The suite runs without torch, so the module must not need it."""

    script = (
        "import sys\n"
        "from living_memory.chunking import chunk_text, get_tokenizer\n"
        "chunks = chunk_text('Проверка ' * 400)\n"
        "assert len(chunks) > 1, chunks\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "assert 'sentence_transformers' not in sys.modules, 'ST was imported'\n"
        "print(get_tokenizer().name)\n"
    )
    root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    env["LIVING_MEMORY_EMBEDDING_BACKEND"] = "hash"

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "heuristic"


def test_hash_embedding_backend_resolves_to_the_heuristic_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TOKENIZER_ENV_VAR, raising=False)
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "hash")
    reset_tokenizer_cache()

    assert get_tokenizer().name == "heuristic"

    monkeypatch.setenv(TOKENIZER_ENV_VAR, "heuristic")
    monkeypatch.setenv("LIVING_MEMORY_EMBEDDING_BACKEND", "auto")
    reset_tokenizer_cache()

    assert get_tokenizer().name == "heuristic"
    reset_tokenizer_cache()


def test_unavailable_model_tokenizer_falls_back_to_the_heuristic_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(TOKENIZER_ENV_VAR, "model")
    for name in ("HOME", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(tmp_path / "st"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "transformers"))
    reset_tokenizer_cache()

    tokenizer = get_tokenizer("no-such-model-anywhere")

    assert tokenizer.name == "heuristic"
    assert chunk_text(REAL_SHAPED_NODE, tokenizer=tokenizer)
    reset_tokenizer_cache()


def test_model_tokenizer_counts_past_the_configured_sequence_limit(
    model_tokenizer: Tokenizer,
) -> None:
    """The model's tokenizer.json ships truncation at 128; counting must ignore it.

    Left enabled, every long text reports exactly 128 tokens, the chunker sees it
    as fitting one window, and the tail stays invisible — the bug chunking exists
    to remove.
    """

    long_text = REAL_SHAPED_NODE

    total = count_tokens(long_text, tokenizer=model_tokenizer)

    assert total > MODEL_MAX_SEQ_LENGTH * 3, total
    assert model_tokenizer.name.startswith("model:")


def test_model_tokenizer_chunks_fit_the_encoder_sequence_limit(
    model_tokenizer: Tokenizer,
) -> None:
    """Re-encoding a chunk on its own must still fit max_seq_length."""

    chunks = chunk_text(REAL_SHAPED_NODE, tokenizer=model_tokenizer)

    assert len(chunks) > 1
    for chunk in chunks:
        standalone = count_tokens(chunk.text, tokenizer=model_tokenizer)
        assert standalone + ENCODER_SPECIAL_TOKENS <= MODEL_MAX_SEQ_LENGTH, chunk


def test_heuristic_tokenizer_approximates_the_model_tokenizer(
    model_tokenizer: Tokenizer, heuristic: HeuristicTokenizer
) -> None:
    """The fallback counter must stay close to — and not below — the real one."""

    real = count_tokens(REAL_SHAPED_NODE, tokenizer=model_tokenizer)
    estimated = count_tokens(REAL_SHAPED_NODE, tokenizer=heuristic)

    assert 0.9 <= estimated / real <= 1.35, (estimated, real)
