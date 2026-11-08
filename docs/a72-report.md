# A7.2 — Competition LLM Mode + Immediate Publication

Дата: **8 ноября 2026**. Ветка: `feature/a72-neuraldeep-production`.

**Verdict: READY FOR ONE AUTHORIZED LIVE TEST в competition mode.** Стандартный
конкурсный запуск с ключом и URL включает A7.1 без trusted adapter. Offline
end-to-end проверен с настоящим openJiuwen 0.1.19 и MockTransport. Реальная Qwen
**не вызывалась**: отдельного разрешения нет. Поэтому фактическое качество Qwen,
совместимость конкретного конкурсного endpoint и реальный billed usage пока
не подтверждены. Предыдущий BLOCKED относился к выпуску strict adapter: для
strict этот вывод сохраняется. Новый режим явно принимает контролируемый риск.

## Изученное и границы доказательств

Изучены Contracts v1, A2 Ledger, A3 Gateway/usage, A3.5 compatibility,
`docs/a35-report.md`, A4, A5 publication, A7.1 после восьми review fixes и CLI.
Contracts v1 и его DTO/Protocols не изменены. Нет второго AgentRunner, второго
независимого Gateway, обхода Ledger или новой coding pipeline.

Первичные публичные источники: [NeuralDeep Chat API](https://neuraldeep.ru/docs/chat.md),
[полная справка](https://neuraldeep.ru/llms-full.txt),
[лимиты и ошибки](https://neuraldeep.ru/docs/limits.md),
[Limits API](https://neuraldeep.ru/docs/limits-api.md).
Документация описывает `-noreason`; примеры `max_tokens` не доказывают hard cap
полного оплачиваемого выхода. Публичные документы не дают достаточных оснований
для verified input bound, total output cap и строгой интерпретации billing usage
конкурсного прокси. Live smoke не восполняет эти гарантии.

Официальные upstream assets существуют, например
[Qwen3.6 tokenizer](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/tokenizer.json)
и [Qwen3.8 repository](https://huggingface.co/Qwen/Qwen3.8-27B/tree/main).
Однако связь конкретных revisions/tokenizer/chat template с NeuralDeep aliases
не подтверждена. Вывод: upstream tokenizer сам по себе не даёт гарантии серверного
счётчика. В этой реализации выбран разрешённый заданием документированный
консервативный estimator. Нет ML-токенизатора, runtime downloads или новых
тяжёлых зависимостей; Docker остаётся offline без ключа.

Для strict по-прежнему нужны ответы провайдера/организаторов:

1. Immutable model/tokenizer/template revisions и верхняя граница всего
   оплачиваемого framing/overhead либо authoritative preflight count/ticket.
2. Поле hard cap всего output, включая скрытый reasoning, внутренние retries,
   timeout/cancel и ошибочные запросы.
3. Нормативная схема prompt/cache/completion/reasoning и её связь с биллингом;
   reconciliation по request ID для неизвестного исхода.

Без этих условий безопасные строгие варианты: offline решения; предоставленный
организаторами счётчик/lease; reviewed adapter для подтверждённого deployment.
Competition — отдельная разрешённая альтернатива с риском недооценки.

## Конфигурация и стандартный запуск

| Конфигурация | Поведение |
| --- | --- |
| Нет `LLM_API_KEY` или ключ состоит из пробелов | Offline B-minimal, ноль HTTP |
| Ключ + `LLM_BASE_URL`, стандартный CLI | Competition Decision Router без adapter |
| `--llm-mode competition` / `LLM_MODE=competition` | Явный competition; ключ всё равно нужен для HTTP |
| `--llm-mode strict` / `LLM_MODE=strict` | Требует reviewed `LLM_COMPATIBILITY_ADAPTER` и verified profiles до HTTP |
| `--components nsu_budget_agent.solvers.composition:build` | Явный offline даже при внедрённом ключе |

CLI-флаг имеет приоритет над environment. Legacy explicit LLM factories сохраняют
strict default; programmatic `build_production` выбирает по RunConfig accounting:
`strict` либо явно заданный `best_effort`. Competition использует существующее
значение `best_effort`, не меняя Contracts. CLI выводит `llm_mode` и
`accounting_mode`, spent/held и decision_routes. Ключи не выводятся.

Default Solver: `qwen3.6-35b-a3b-noreason`. `LLM_MODEL` выбирает Solver;
`LLM_REVIEW_MODEL` выбирает Reviewer, по умолчанию совпадает с Solver.
Оба допускают все четыре конкурсных ID: qwen3.8-27b, qwen3.8-27b-noreason,
qwen3.6-35b-a3b, qwen3.6-35b-a3b-noreason. Выбор происходит для каждого запроса
по роли; неверный ID отклоняется. Сложного model optimizer нет.

## Резервирование, usage и остановка

Правила конкурса приняты из задания: `input / 10 + output`, весь reasoning —
output, cache не даёт скидки. A2 хранит `input + 10 * output` Units.
`BUDGET=100000` означает **1000000 Units**; float-арифметики нет.

В `llm/competition.py` input estimate равен **числу байт ASCII-escaped JSON body
+ 1024 + 256 на сообщение**. Body включает модель, роли, content, output cap и
framing JSON. Это запас по сравнению с content/4, но **не доказанная верхняя
граница** неизвестного серверного framing. Все ProviderProfile flags остаются
false; trusted adapter не импортируется и не подделывается.

`LLM_OUTPUT_CAP` по умолчанию 512, допустимо 64..2048. Перед каждым HTTP Gateway
берёт минимум свободного глобального и attempt budget, уменьшает output cap,
повторно сериализует body и резервирует estimate + 10 * cap в A2. Если минимум
64 не помещается, HTTP отсутствует. Оплата/резерв не обходят A2/A3.

A4 сначала оценивает весь портфель, решает бесплатные задачи, затем выбирает
платные по reward с floor/equal-share policy. Competition preflight проверяет
модели обязательных ролей и подбирает равные caps между 64 и настроенным максимумом,
чтобы оценки обязательных запросов поместились в доступный бюджет. Если даже
минимальные caps не помещаются, задача откладывается до первого HTTP. После чтения
источников preflight повторяется. Это оценка маршрута, а не резервирование будущего
неизвестного prompt: A2 резервирует только фактический запрос, A3 сериализует и
оценивает каждый запрос заново. Более длинный следующий prompt может не поместиться;
низкий cap может дать незавершённый ответ. Legacy strict policy сохранена.

Usage нормализуется отдельно от строгой policy:

- Integer prompt/completion считаются полностью, cached_tokens не вычитается.
- При явном reasoning=0 без reasoning text согласованный total либо его
  отсутствие позволяет settled reported counters.
- При положительном reasoning и total=P+C reasoning уже включён в C (R<=C).
  При total=P+C+R reasoning добавляется ровно один раз.
- Для `-noreason` без reasoning counters/text нужны P, C и total=P+C.
- Missing, ambiguous, противоречивые counters, reasoning text с нулевым/неизвестным
  reasoning, ошибочный HTTP и потерянный ответ дают **unknown**, cost=None;
  резерв остаётся **held**, дальнейшие платные маршруты останавливаются.

`UsageRecord.status=exact` здесь означает однозначное арифметическое settlement
**сообщённых** counters по opt-in policy, а не verified remote billing.
Успешный smoke не повышает профиль. Отдельные reasoning/cached counters никогда
не считаются скидкой или вторым списанием одних и тех же токенов.

Timeout/unknown, quota errors, 402, 429 и 500 не запускают автоматический retry.
При сообщённом превышении estimate/cap полный reported расход записывается,
Ledger блокирует новые платные запросы; превышение не обрезается до резерва.
При успешном HTTP и однозначном usage полный ответ передаётся обычному Router и
Validator. После перерасхода публикация требует независимой проверки: review-only
семантика не становится сертифицированной, а заблокированный Reviewer не обходится.
`finish_reason=length`, неверный ответ и unknown usage не публикуются.
Известные quota codes и 402 останавливают paid routes; бесплатные поддерживаемые
маршруты доступны при оставшемся времени. Серверная остановка квоты не считается
доказательством точной стоимости отдельного запроса.

## Немедленная публикация и crash proof

Publisher не переписывался. Если deadline попытки истекает после acceptance,
принятый ответ всё равно передаётся A5; новые платные маршруты этого run закрываются.
Фиктивный `asyncio.timeout(5)` удалён: он не ограничивал ожидание `joined_io` и
мог скрыть failed receipt после задержанной ошибки файловой системы. После
acceptance A5 ожидает завершения publication перед очисткой staging и переходом
к другой задаче; фоновой публикации нет. Системный вызов файловой системы,
зависший в ядре, нельзя безопасно прервать из Python-потока: гарантированного
верхнего предела времени ожидания нет. Это внешний риск зависания процесса;
принудительное завершение извне может оборвать ещё не опубликованный ответ.
AgentRunner ожидает Validator, acceptance и полностью завершённый publish до
следующей задачи. Publisher пишет уникальный temp в каталоге задачи, делает
flush/fsync файла, проверяет snapshots/identity, затем атомарно публикует
`answer.txt` через **no-clobber hard link**. Это сохраняет атомарную видимость и
отказ от перезаписи существующего ответа. Temp удаляется до возврата; на POSIX
есть best-effort directory fsync. Публикация не откладывается в shutdown run.

Тест первой задачи получает неизвестную Python-грамматике инструкцию, проходит
mock Qwen Solver → Reviewer → Validator → A5. Перед callback HTTP второй задачи
проверяются полные bytes первого answer и выполненный fsync. Затем отдельно
возвращаются timeout/402/429/500: предыдущий answer неизменен, повторного HTTP нет.

`a72_crash_worker.py` запускает настоящий CLI/openJiuwen в отдельном процессе,
использует только MockTransport, публикует бесплатный ответ 5 и принятый LLM
ответ 15, входит в следующий mocked HTTP и пишет fsynced marker. Родитель вызывает
`process.kill()` — TerminateProcess на Windows / SIGKILL на POSIX. После kill оба
answer.txt целы, у прерванной задачи ответа нет, маркер `finally` отсутствует.
Это проверяет сохранность при принудительном завершении без exception handler.
Это **не** доказательство сохранности при отключении питания любой ОС/ФС:
на Windows directory fsync в данном publisher недоступен.

## Offline проверки

Все интеграции используют настоящий SDK, A2/A3/A4/A5 и запрещённый real
AsyncHTTPTransport. Test-only attestations относятся только к strict mocks.

| Сценарий | Фактический ожидаемый результат |
| --- | --- |
| Default keyed CLI, неизвестная инструкция, без adapter | answer=15; 2 HTTP; 1200 Units; held=0; profile flags=false |
| Strict CLI с неподтверждённой/неполной конфигурацией | Ноль HTTP; ошибка до отправки |
| Без ключа | Offline answer=5; spent=held=0 |
| Included reasoning P=1000, C=100, R=40, cache=900 | 2000 Units, output=100 |
| Separate reasoning, total=1140 | 2400 Units, output=140 |
| Missing / ambiguous usage | 1 HTTP; spent=0, held>0; ответа нет; paid blocked |
| Reported cap breach C=600 при cap=512, неизвестная семантика | 1 HTTP; 6120 Units учтено; paid blocked; обязательный Reviewer недоступен, ответа нет |
| 8 free + 4 paid разных rewards | 12 ответов; paid order 100,60,40,10; 8 HTTP; 4800 Units |
| Нельзя зарезервировать обязательный review | Ноль HTTP; spent=held=0 |
| Cap 2048 при available=6500, независимо проверяемый affine | Cap уменьшается до отправки; answer=4 |
| Timeout/402/429/500 после первой публикации | Первый answer=15 сохранён; 3 HTTP всего; 1200 spent + held неизвестного запроса |
| Kill subprocess в следующем HTTP | Предыдущие answers 15 и 5 целы; finally не выполнен |

## Исправления независимого review

| Дефект | До | Исправление и регрессия после | Оставшийся риск |
| --- | --- | --- | --- |
| P1: перерасход отбрасывал ответ | Полный affine `4` пропадал при 4480, 12480 или 6120 Units | `test_excess_reported_usage_keeps_independently_verifiable_answer`: полный charge, halt и независимая проверка с публикацией; отдельные тесты запрещают обход Reviewer, неверный и незавершённый ответ | Unknown usage удерживается; без независимой проверки публикации после overrun нет |
| P1: deadline после acceptance | `passed/accept`, но answer отсутствовал | `test_accepted_answer_publishes_after_attempt_deadline_without_paid_followup`: answer=5, следующий платный вызов не начат | Принудительное завершение до атомарной публикации может прервать запись; зависший системный I/O невозможно жёстко ограничить без нарушения безопасности |
| P2: cap floor приводил к defer | 10000 Units и cap 512: ноль HTTP; вручную cap 64: answer=15 | `test_preflight_negotiates_caps_for_mandatory_review`: cap подбирается автоматически, 2 HTTP, 1200 Units | Размер будущего prompt неизвестен, cap может оказаться недостаточным для полного ответа |
| P2: Reviewer вне allowed_models | Solver выполнял HTTP и тратил 600 Units до отказа Reviewer | `test_forbidden_reviewer_is_checked_before_solver_http`: ноль HTTP и расхода; allowed_models не расширяется | Другие роли могут понадобиться после непредвиденного предложения модели; перед их HTTP действует проверка Gateway |

Финальный re-review также установил, что прежний пятитсекундный timeout не
прерывает заблокированный `fsync`. Теперь timeout удалён. Регрессия
`test_blocked_fsync_keeps_staging_and_publishes_once_before_next_task` держит
`fsync` более пяти секунд и проверяет сохранность staging, отсутствие receipt,
ответа и следующего HTTP до освобождения; затем проверяет ровно один saved receipt,
одну атомарную публикацию и завершённую очистку. Отдельный тест
`test_delayed_fsync_failure_returns_real_failed_receipt` воспроизвёл потерю failed
receipt до изменения и подтвердил корректную обработку ошибки после него.
Attempt-deadline, no-clobber, crash и cancellation тесты сохранены.

Новые тесты: `tests/unit/test_competition_usage.py`,
`tests/integration/test_a72_competition.py`,
`tests/integration/test_a72_production.py`; test child —
`tests/integration/a72_crash_worker.py`. Сохранён stress A7.1 на 100 задач с
резервом 13312, приоритетом верхних 25 и учётом 200 вызовов A2.
Дополнительный изолированный A4/A2 stress competition на 100 задач проверил
порядок всех rewards, включая верхние 25: 200 смоделированных списаний,
120000 Units, held=0; HTTP и реального SDK в этом stress нет.

Итоговые проверки Windows / Python 3.11:

| Проверка | Результат |
| --- | --- |
| Адресный набор publication/deadline/crash | 20 тестов, OK, 31,207 с; отдельный тест отмены публикации — OK |
| Полный unittest discovery после удаления timeout | **334 теста, OK (skipped=6), 183,827 с**; выполнены 328, платформенных пропусков 6 |
| Ruff lint / format-check | Успешно; 93 Python-файла отформатированы |
| Mypy, CI scope и orchestration/decision.py | Без ошибок в 59 source files |
| Host smoke, переносимые задачи | 3/6, 7 SDK workflows, spent=held=0 |
| Host smoke, публичные 12 задач | 6/12, 12 SDK workflows, spent=held=0; нерешённые без answer.txt |
| git diff --check | Успешно |
| Docker / Podman | Команды отсутствуют; build/run не выполнялись |
| Реальная Qwen | Не вызывалась; live acceptance остаётся открытым |
| Scripted benchmark A7.1 | 6/8 правильных, 0 ошибочных, 11 mock HTTP, 6600 Units; без изменения |

Добавлено 7 регрессионных тестов к прежним 325; до исправлений четыре находки
воспроизводились постоянными тестами. После re-review добавлены два теста
заблокированного и ошибочного `fsync`, доведя общее число до 334. Ранее regression test
preflight воспроизвёл отправку при недостаточном резерве обязательного review;
после исправления HTTP нет. Увеличение preflight для legacy strict меняло прежнее
распределение бюджета, поэтому новая консервативная policy включена только в
competition; прежний strict replan/allocation regression снова проходит.

Воспроизведение из корня проекта:

```powershell
$env:PYTHONPATH = 'src'
$env:MYPYPATH = 'src'
.\.venv-a1\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-a1\Scripts\python.exe -m ruff check src tests run_agent.py
.\.venv-a1\Scripts\python.exe -m ruff format --check src tests run_agent.py
# Mypy: scope .github/workflows/a1-ci.yml + orchestration/decision.py.
.\.venv-a1\Scripts\python.exe tests/docker/smoke.py --mode host
.\.venv-a1\Scripts\python.exe tests/docker/smoke.py --mode host --source C:\VSCode\openJiuwen\nsu_benchmark_smoke_12_tasks\tasks
```

Журналы: `%TEMP%/a72-io-full.log`, `a72-io-target.log`, `a72-io-cancel.log`,
`a72-io-host.log`, `a72-io-public.log`. Smoke создаёт свежие
временные копии; исходный публичный набор не изменялся. Docker smoke подготовлен
в существующем CI, но локальный host smoke не заменяет Linux Docker run.

## Оставшиеся риски и первый реальный запуск

Неизвестный server overhead, игнорируемый cap, скрытый неотражённый reasoning,
неверный usage или продолжение генерации после timeout могут превысить estimate.
Режим не гарантирует hard upper bound реального расхода. A2 держит unknown, но
его состояние в памяти не переживает kill; нельзя автоматически начинать новый
paid run со старой квотой без reconciliation. Уже опубликованные ответы этого
не требуют. LLM review не доказывает правильность arbitrary direct_answer;
сохраняются ограниченная review-policy A7.1 и независимые локальные проверки.
Новые файлы не расширяют source allowlist, инструментальные полномочия или A5
acceptance. Неизвестные или небезопасные инструкции по-прежнему могут остаться
без ответа. Deadline/call caps ограничивают попытки, не гарантируют reward.

После **отдельного разрешения** на один реальный тест вне Python-грамматики
из корня проекта, в окружении с установленными dependencies:

```powershell
# LLM_API_KEY и LLM_BASE_URL заранее предоставлены организаторами.
# Не вставлять ключ в команду/отчёт. Каталог содержит одну свежую задачу.
$env:BUDGET = '100000'
$env:LLM_MODE = 'competition'
$env:LLM_MODEL = 'qwen3.6-35b-a3b-noreason'
$env:LLM_REVIEW_MODEL = 'qwen3.6-35b-a3b-noreason'
$env:LLM_OUTPUT_CAP = '512'
.\.venv-a1\Scripts\python.exe run_agent.py --tasks-dir <fresh-one-task-dir> --llm-mode competition --timeout-s 120
```

Acceptance live: правильный answer.txt после Validator/A5, одна из четырёх
реальных Qwen, recorded reported usage либо честный unknown/held, сравнение с
доступным provider billing. Unknown не закрывает требование учтённого расхода.
Эта команда **не выполнялась**. Commit/push/merge не делались.
