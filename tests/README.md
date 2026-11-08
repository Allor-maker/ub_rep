# Проверки

Реализованы 27 проверок C01, 24 бюджетных C04–C06, 4 проверки конфигурации
и 35 mock HTTP проверок A3 в `unit/test_llm_gateway.py`, а также 51 проверка A4
(всего A1–A4: 141). A5-lite добавляет 33 теста, включая реальный SDK workflow
и отдельный CLI-процесс. B-minimal добавляет 30 тестов, A6 — тест конкурсного CLI:
всего 205, в текущем Windows
шесть платформенных пропусков. Три внешних dataset-проверки требуют локальных наборов;
переносимые production-fixtures запускаются всегда. См. [отчёт B](../docs/b-minimal-report.md).
Docker smoke harness и статус Linux-проверки: [отчёт A6](../docs/a6-report.md).
Полный suite требует Python 3.11, HTTPX и openJiuwen 0.1.19; C01 проверяет импорт с `-S`.
Из корня репозитория в PowerShell:

```powershell
python -m venv .venv-a1
.\.venv-a1\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-llm.txt -r requirements-sdk-probe.txt
$env:PYTHONPATH = 'src'
.\.venv-a1\Scripts\python.exe -m unittest discover -s tests -v
```

Статический пример реализаций B — `contracts/typecheck_consumers.py`.
Команды mypy/ruff и отдельного SDK smoke, результат Windows-проверки и открытая
Linux-проверка описаны в [отчёте A1](../docs/a1-report.md). Команды проверки ledger
и границы C04–C06 — в [отчёте A2](../docs/a2-report.md). Остальные сценарии ещё предстоят.
Gateway и связанные Protocol проверяются также через `contracts/typecheck_llm.py`;
команды и ограничения — [отчёт A3](../docs/a3-report.md). Реальная сеть в HTTP-тестах запрещена.
Сквозные проверки, обязательные зависимости и команды A5-lite — в
[отчёте A5](../docs/a5-report.md). Fake B существуют только в `integration/fakes.py`.

| Каталог | Назначение | Ответственный |
| --- | --- | --- |
| `contracts/` | Совместимость объектов, ошибок и расширений | A + B |
| `unit/` | Бюджет, загрузка, отдельные инструменты и проверки | Владелец компонента |
| `integration/` | Сквозные связи, включая openJiuwen и публикацию | A + B |
| `acceptance/` | Сборка/запуск, отбор, лимиты и сохранность результатов | A; независимая приёмка B |
| `fixtures/` | Малые безопасные собственные данные | B |

Обычные проверки автономны и не обращаются к NeuralDeep. Тестовые реализации
транспорта/решателя допустимы здесь, но не маскируют отсутствие runtime-реализации.
Реальные API и Docker-проверки добавляются как явно выделенные сценарии с
документированными предпосылками; пропуск такого сценария не означает его успех.

Минимальная матрица: неизвестный/отсутствующий type, новые имена/форматы, workspace
есть/нет, утечка expected через чтение, неподдерживаемость и ошибка, отмена,
параллельный бюджет, usage/reasoning, timeout без usage, rate limit, атомарная
публикация, неизменность тестов, отсутствие runtime-загрузок, две отборочные задачи.

## A7.1

`unit/test_a71_decision.py` проверяет таблицу переходов, единственное владение
состоянием, fingerprint запросов и JSON-контракт Reviewer.
`integration/test_a71_router.py` запускает настоящий SDK, A2/A3 MockTransport,
инструменты, Validator и A5. Проверяются частичные кандидаты, review/replan,
неизвестные инструкции, tool_plan/direct_answer/decline, независимые вычисления,
ошибки JSON/evidence, инъекции, held usage, deadline и бюджет всего портфеля.
Скриптовое сравнение B-minimal/A3.6/A7.1 не измеряет понимание языка моделью;
метрики и риски подробно приведены в [отчёте A7.1](../docs/a71-report.md).
Исправления независимого review проверяются в `integration/test_a71_review_fixes.py`,
`unit/test_a71_allocation.py` и `unit/test_a71_certification.py`: source privacy,
отвергнутые кандидаты, независимые вычисления, run isolation, v1 stop и 100 задач A4/A2.

## A7.2

`integration/test_a72_production.py` проверяет стандартный CLI без `--components`:
offline без ключа, strict Router с mock-attestation и MockTransport, отказ до
transport при неполной конфигурации или любом неподтверждённом профиле, явный
offline при наличии ключа. Все provider profiles относятся только к mock.
Пройденный scripted end-to-end не подтверждает реальные гарантии NeuralDeep;
статус production и недостающие условия — в [отчёте A7.2](../docs/a72-report.md).
Этот же файл проверяет competition по умолчанию без adapter и явный strict.
`integration/test_a72_competition.py` проверяет настоящий SDK + A2/A3/A4/A5:
разные Qwen ID, unknown/held, output cap, порядок reward, немедленную публикацию
перед timeout/402/429/500. `a72_crash_worker.py` — только test subprocess:
родитель принудительно завершает его внутри следующего mocked HTTP и проверяет
сохранённые answer.txt без выполнения finally. Реальный transport запрещён.
`unit/test_competition_usage.py` проверяет Units, cache и included/separate reasoning.

## A7.3

`integration/test_a73_quality.py` tests changed public parameters, independent
math/CSV/Markdown verification, negative conditions, source scope and safe usage
events. `unit/test_cli_deadline.py` separates attempt and run deadlines.

Offline 20-task replay (no key, real SDK with network-denying MockTransport):

```powershell
$env:PYTHONPATH = 'src;.'
.\.venv-a1\Scripts\python.exe -m tests.benchmarks.a73 --mode offline --source C:\VSCode\openJiuwen\live-test\tasks --grader C:\VSCode\openJiuwen\live-test\grade.py
```

Harness exports fresh public inputs under `experiments/runs/`; the external
grader runs after agent exit. Mock results are not Qwen results. Live modes need
explicit authorization and record one smoke/full allowance in a local guard.
The A7.3 smoke allowance has been used; its failure means another live run needs
new authorization. See [A7.3 report](../docs/a73-live-quality-report.md).

## A7.4

`integration/test_a74_reliability.py` runs 45 SDK scenarios: changed names, files,
values and condition ordering, canonical/workspace paths, and malformed or
semantically incomplete Search/CSV plans. A forged Reviewer accept cannot rescue
a supported but mismatched plan. `unit/test_a74_guard.py` tests durable cumulative
accounting, reservation of the next call, unknown usage/timeouts and the HTTP cap.
`unit/test_a74_lcm.py` checks nested bounded LCM with changed operands and rejects
invalid arity, zero/negative/fractional/oversized operands, keywords and arbitrary calls.

`benchmarks/a74.py` is an explicitly invoked live diagnostic harness, never part
of unittest discovery. It uses the real competition Gateway and SDK. Its single
session journal is `experiments/runs/a74/journal.json`; do not delete/reinitialize
it to obtain a fresh allowance. The session is authorized for at most 25 HTTP
requests (quota/documentation GETs included conservatively) and 100000 reported Units in total.
An exclusive lock prevents concurrent experiments; a stale lock or durable held
amount requires investigation, never automatic release/restart. Estimates do
not guarantee provider billing. No live request is made by the offline tests.
Raw diagnostic files stay in the ignored, ACL-restricted session directory.
See [A7.4 report](../docs/a74-llm-reliability-report.md) for results and limitations.

## A7.5 recovery

`integration/test_a75_recovery.py` covers parameterized inventory operations,
record/quantity distinction, status-set predicates, original-hash Markdown
excerpts, corpus bounds/distractors/injection, unsupported qualifications,
controller-option isolation and one affordable output-length retry. Parse caches
store immutable encodings and reconstruct fresh ToolCalls; source and answer
validation are never cached. `unit/test_a75_session.py` verifies cumulative
500000-Unit accounting, restart/unknown/error holds, cap breaches and redaction.

`benchmarks/a75.py` is explicitly invoked; unittest never enables real network.
It exports only public inputs from `live-test/tasks2`, never reads a grader or
gold, and uses the real CLI/SDK/Gateway. Its exclusive lock and durable journal
under `experiments/runs/a75/` span all subprocesses. Live mode requires a readable
configured dotenv alias or LLM environment variables, checks provider quota and
never overwrites an earlier experiment. Do not delete/reset the journal to
obtain another allowance. A7.5 opts into `LLM_REQUIRE_INDEPENDENT=1` and
`LLM_LENGTH_RETRY=1`; legacy defaults remain compatible. See
[A7.5 report](../docs/a75-benchmark-recovery-report.md).

## A4

`unit/test_task_loader.py`, `unit/test_planner.py` и `contracts/test_a4_planning.py`
проверяют read-only загрузку, безопасный ввод, реестр, чистое планирование и coverage
с fake-компонентами B. `contracts/typecheck_planning.py` проверяет неизменные порты.
Полный discovery CI включает новые тесты автоматически; POSIX и symlink проверки
могут быть пропущены на Windows и должны выполниться в Linux CI.
