# Деплой-хендофф: леджер transcript-grounding (фаза 3)

Оператору. Этот релиз — единственная точка цели transcript-grounding,
требующая деплоя и рестарта LM. Сам узел ничего не деплоил и не
рестартовал; всё ниже — операторские действия.

## Что деплоить

Ветка `transcript-grounding--ledger-schema` (коммит с этим документом; после
мержа — соответствующий merge-коммит в master). Изменения:

| Файл | Что в нём |
|---|---|
| `src/living_memory/storage.py` | `TRANSCRIPT_GROUNDING_SCHEMA_SQL` (CREATE-only DDL) + один `executescript` в `_initialize_schema` |
| `src/living_memory/transcript_ledger.py` | новый модуль: парсинг verdict-JSONL, идемпотентный импорт |
| `scripts/transcript_ledger_import.py` | офлайн CLI импорта |
| `tests/test_transcript_ledger.py` | пины: байт-идентичность DDL, реоткрытие, идемпотентность, чистота живого пути |

## Что меняется в базе

Одна новая таблица `transcript_grounding_verdicts` + один индекс
`idx_transcript_grounding_node_graded`. Больше ничего:

- `SCHEMA_VERSION` остаётся **8** (аддитивно, как recall-delivery-history);
- DDL всех существующих таблиц (включая `nodes`/`connections`/`recall_events`
  живой 500-МБ базы) — **байт-в-байт** без изменений; тест сравнивает
  `sqlite_master` свежей базы текущего билда против билда master;
- никаких бэкфиллов для новой таблицы — она рождается пустой;
- ключ идемпотентности `UNIQUE(recall_event_id, node_id, method_version)`:
  повторный импорт того же файла — replay, не дубли;
- FK нет намеренно (прецедент `recall_lookup_events`): вердикты грейдятся
  офлайн по снапшоту, к моменту импорта событие может быть вычищено, а
  `field='alt'` несёт id чужого стора. Колонка `field` (`local|alt`)
  говорит, чьим `recall_events` принадлежат id строки.

## Почему нужен рестарт

Таблица создаётся внутри `MemoryStore._initialize_schema`, который
выполняется **при открытии файла**. Живой сервер держит старый код и уже
открытый стор — деплой кода без рестарта не меняет в живой базе ничего.
Рестарт и есть миграция: первое открытие новым билдом добавляет таблицу
(`CREATE TABLE IF NOT EXISTS` на каждом открытии, отдельного
`_migrate_pre_v9` нет — паттерн v8/`recall_attestations`).

Рестарт безопасен:

- ни один живой путь (recall / remember / teach / lookup) таблицу не читает
  и не пишет — доказано трейсом всех SQL-стейтментов живых тулов в
  `test_live_recall_remember_paths_never_touch_the_ledger`;
- стоимость на открытии — два `CREATE ... IF NOT EXISTS`, без пересборки
  чего-либо;
- откат = вернуть предыдущий билд. Таблица останется в файле и это
  безвредно: старый код о ней не знает, ничто её не читает. Дропать не
  требуется.

## Порядок (симметрично на обоих хостах: local и alt)

1. Задеплоить код.
2. Рестартовать LM-сервер в спокойный момент.
3. Верифицировать (ниже).
4. Импортировать вердикты фазы 2, когда они появятся
   (`artifacts/transcript-grounding/grade/…`, per-host).

## Верификация post-restart

```bash
DB=~/.local/share/living-memory/global.sqlite3   # на alt — его путь

# 1. Оба новых объекта на месте:
sqlite3 "file:${DB}?mode=ro" \
  "SELECT name FROM sqlite_master WHERE name IN
   ('transcript_grounding_verdicts','idx_transcript_grounding_node_graded');"
# ожидание: обе строки

# 2. Версия схемы не двинулась:
sqlite3 "file:${DB}?mode=ro" \
  "SELECT value FROM metadata WHERE key='schema_version';"   # 8

# 3. Целостность:
sqlite3 "file:${DB}?mode=ro" "PRAGMA quick_check;"           # ok

# 4. Таблица пуста до импорта:
sqlite3 "file:${DB}?mode=ro" \
  "SELECT COUNT(*) FROM transcript_grounding_verdicts;"      # 0
```

Плюс смоук через MCP: `memory_health` и любой `memory_recall` отвечают как
до рестарта. Опционально до рестарта — прогнать
`scripts/verify_live_db_migration.py <DB>` (копия через backup API,
миграция на копии, integrity + счётчики; живой файл не трогается).

## Импорт вердиктов

```bash
# сначала посмотреть, что будет сделано (ничего не пишет, даже DDL):
python scripts/transcript_ledger_import.py --db "$DB" \
    --verdicts artifacts/transcript-grounding/grade/verdicts-local.jsonl --dry-run

# затем сам импорт (одна транзакция; прерванный прогон не оставляет следов):
python scripts/transcript_ledger_import.py --db "$DB" \
    --verdicts artifacts/transcript-grounding/grade/verdicts-local.jsonl \
    --json artifacts/transcript-grounding/ledger/import-report-local.json
```

- Коды выхода: `0` — импортировано/реплеено чисто; `1` — найдены
  mismatched-строки (тот же ключ, другие числа — нарушение дисциплины
  `method_version`; в базу НЕ применены, перечислены в отчёте); `2` —
  битый вход или недоступная база, ничего не закоммичено.
- Повторный запуск того же файла всегда безопасен (replay).
- Пере-грейд легитимен только под новым `method_version` — ляжет рядом со
  старыми строками, историю не перепишет.
- Импортёр не конструирует `MemoryStore` — миграции/бэкфиллы стора из него
  не запускаются; он умеет создать таблицу и на файле до рестарта (тот же
  общий DDL-констант), но рекомендуемый порядок — деплой → рестарт →
  верификация → импорт. При живом сервере импорт допустим
  (busy_timeout 30 с), лучше в спокойный момент.
- Каждый хост импортирует свой файл вердиктов (`field` в строках должен
  соответствовать хосту прогона фаз 1–2).

## Чего в этом релизе нет

Никакого подкрепления от транскрипт-сигнала: чтение леджера живым путём —
фаза 5 (env-клапан, по умолчанию выключен), отдельный релиз по числам
replay A/B фазы 4.
