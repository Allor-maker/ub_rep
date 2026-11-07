# A5-lite — минимальный сквозной запуск

Реализовано в `feature/a5-orchestration`. Контракты A1, интерфейсы Ledger A2,
Gateway A3, Planner A4 и компоненты B не изменены. A6/A7 не реализованы.
NeuralDeep не вызывался; commit/push/merge не выполнялись.

## Что работает

`run_agent.py → DirectoryTaskLoader → BaselinePlanner → JiuwenAdapter → Solver.step
→ Validator → V1AcceptancePolicy → AnswerOnlyPublisher`.

- `orchestration/runtime.py`: доверенная сборка `Components`, последовательный
  драйвер задач, оценки SolverOffer, ограниченные попытки, ожидание not_before,
  состояния pending/running/deferred/exhausted/submitted. `None` от планировщика
  проверяется через `next_not_before_wakeup`; временно отложенная первая попытка
  не превращается в окончательное завершение. Без временного пробуждения/активной
  попытки драйвер завершает текущий запуск; фонового ожидания позднего usage нет.
  Исключение `assess` изолируется для одной пары task/solver: её offer не создаётся,
  остальные пары продолжают оцениваться. `RunReport.assessment_errors` содержит
  task_id/solver_id и безопасную ErrorInfo без текста исходного исключения;
  CLI выводит число assessment_errors. Отмена, KeyboardInterrupt и SystemExit
  распространяются. Ошибка оценки не считается начатой попыткой.
- `orchestration/engine.py`: **настоящий openJiuwen 0.1.19** строит и исполняет
  Workflow с Start, цепочкой ограниченных step-компонентов и End. Каждый активный
  компонент вызывает один Solver.step и максимум один ToolExecutor.execute.
  После Candidate/Stop оставшиеся компоненты не вызывают решатель. SDK retries=0;
  весь workflow ограничен дедлайном попытки. В состояние SDK передаётся только
  счётчик tick, а не содержимое задач, ответов и ключей.
  Timeout при открытии уже просроченного AttemptBudget возвращает deadline,
  не создаёт staging и не запускает workflow.
- Ledger открывает/закрывает попытку; в SolverContext передаётся сессия Gateway,
  дополнительно запрещающая более одного complete за шаг. Новый баланс не создаётся,
  повторного списания usage в оркестраторе нет. Резервирование до HTTP остаётся
  обязанностью неизменённого A3. Диагностический usage_request_ids содержит ID
  успешных LLMReply; неизвестные отправки без reply остаются в Ledger, который
  является источником истины, в том числе после close_attempt и отмены.
- `task_io/workspace.py`: отдельный временный staging для каждой попытки,
  копирование только разрешённых ArtifactRef через безопасные открытия A4,
  проверка исходных размеров/SHA-256, запрет reserved/symlink/reparse/hardlink
  путей, лимиты обхода и общего объёма. task.json, expected и старый answer.txt
  не попадают в staging. Loader разбирает JSON manifest, но поле expected не
  извлекается и не передаётся решателю. На исходные данные инструменты не направляются.
- После кандидата вычисляются diff и snapshot по Contracts v1. Protected-файлы
  сверяются с исходными хешами, включая защиту, добавленную инструментом;
  evidence_paths должны существовать. Файлы frozen staging переводятся в read-only,
  целостность повторно проверяется хешами после валидации и перед публикацией.
  Валидатор получает **отдельную проверенную копию** и может использовать её для
  временных файлов; frozen staging сохраняется. Создание/очистка копий и запись
  результата дожидаются завершения своих I/O workers при отмене.
- `submission/policy.py`: совпадение attempt/snapshot, четыре обязательных
  blocking-check, отсутствие failed и прохождение всех blocking-check.
  Nonblocking inconclusive не маскирует отсутствие семантического доказательства.
  B Validator отвечает за смысл check_hint, обнаружение тестов и blocking
  visible_tests, если они необходимы; production-валидатор здесь не подменяется.
- `submission/publisher.py`: точные UTF-8 байты CandidateAnswer, без strip и
  дополнительного перевода строки. Повторно проверяются frozen manifest, пустой
  diff и исходные артефакты. Запись во временный файл рядом с task.json, flush/fsync,
  затем атомарное создание answer.txt без перезаписи через hard link; при поддержке
  выполняется fsync каталога. Старый answer.txt — conflict. Внутри запуска журнал
  даёт already_saved для той же попытки/снимка и conflict для другой. submitted
  выставляется после saved/already_saved, включая отмену во время завершения записи.
  При повторе также сверяется SHA-256 точных UTF-8 байтов переданного текста:
  тот же attempt/snapshot с другим текстом возвращает conflict.
- `task_io/source_guard.py`: одна процессная блокировка для доверенных операций
  агента с исходными каталогами, общая для экземпляров publisher. Копирование
  исходников и весь commit используют её в синхронных I/O workers; блокировка
  не удерживается непосредственно в event loop через await. Сейчас единственный
  собственный писатель исходного дерева — answer-only publisher; B работает
  только в staging. Любая будущая собственная запись в оригиналы обязана использовать
  тот же guard и соблюдать терминальность submitted. Сначала проверяются staging
  и исходный манифест. После записи/flush/fsync временного ответа, до hard link,
  проверки повторяются. Проверяется полный набор разрешённых исходных файлов,
  их размеры и SHA-256, включая добавление/удаление файлов. Manifest/expected,
  служебные имена и старые ответы исключаются из обхода без чтения содержимого.
  При обнаружении изменения исходников — conflict без answer.txt; при нарушении
  frozen staging — отказ без публикации. Идентичность task_root проверяется перед commit.

## Запуск и подключение B

Нужны Python 3.11, HTTPX и установленный `openjiuwen==0.1.19` из существующих
requirements. Новых runtime-пакетов не добавлено. Это ещё не финальная сборка A6;
`pip install .` не поддерживается.

```powershell
$env:BUDGET = '100.5'
python run_agent.py --tasks-dir <path> --budget-unit competition_tokens --components my_package.composition:build
```

`build(ledger, config)` возвращает `orchestration.runtime.Components` с реестром
Solver, ToolExecutor, Validator и LLMGateway. Это доверенный модуль приложения,
явно выбранный оператором; задачи не управляют импортами. Factory должна передать
Gateway тот же Ledger и strict-параметры RunConfig. Реальные B ещё отсутствуют:
без `--components` CLI завершится с кодом 2 и не создаст ответы. Программный вход:
`AgentRunner(...).run(tasks_dir)`; после работы требуется `await runner.aclose()`.

### Требования к production factory и конкурсной команде (B/A6)

Команда `python /app/run_agent.py --tasks-dir /data/tasks` **сейчас завершается
с ошибкой argparse**, поскольку не задан обязательный --budget-unit. Одного BUDGET
и LLM_BASE_URL/LLM_API_KEY недостаточно: без --components CLI также отказывает.
Это открытый интеграционный вопрос, а не поддерживаемый конкурсный режим A5-lite.
Неподтверждённая единица и fake B по умолчанию не добавлялись.

Для B/A6 требуется:

1. Подтвердить масштаб и источник BUDGET. Согласовать передачу единицы и выбор
   production factory без дополнительных аргументов конкурсной команды — например,
   через заранее согласованную доверенную конфигурацию сборки/окружения. Такой
   fallback пока не реализован; нельзя молча выбирать competition_tokens.
2. Установить доверенный module:factory вместе с приложением. Синхронная
   `build(ledger, config)` вызывается внутри активного event loop и возвращает
   Components с настоящими SolverRegistry, ToolExecutor, Validator и Gateway.
   Registry содержит ровно одного generic; deterministic_solvers указывает только
   маршруты, действительно не использующие LLM. Factory не читает конфигурацию
   из задач, не импортирует tests.integration.fakes, не подставляет ответы;
   тестовый пакет не включается в production-сборку.
3. Создать NeuralDeepGateway в том же loop с переданным Ledger и strict-режимом
   RunConfig; передать LLM_BASE_URL/LLM_API_KEY, подтверждённый input_bound с framing,
   проверенные ProviderProfile для используемых Qwen, семантику reasoning и полного
   output cap, согласованные concurrency/rate/timeouts. До подтверждения профилей
   strict остаётся fail-closed; verified-флаги не выставляются ради запуска.
4. Реальный ToolExecutor должен обеспечить ОС-изоляцию, ограничения ресурсов и
   остановку дочерних процессов; Validator — обязательные проверки Contracts v1
   и blocking visible_tests, когда они нужны. Оригиналы доступны только publisher,
   исходные изменения по-прежнему unsupported в A5-lite.
5. Определить владельца и гарантированное async-закрытие ресурсов Gateway/инструментов
   при успехе, ошибке и отмене. Текущий runner.aclose останавливает SDK; автоматического
   закрытия произвольных B-ресурсов по Protocol нет. В программной сборке Gateway
   можно держать во внешнем async context; production CLI-сборку ещё нужно связать
   с таким lifecycle без изменения Contracts v1.
6. Подготовить Linux runtime с Python 3.11, SDK 0.1.19 и согласованными зависимостями,
   доступным для записи /tmp и каталогом задач. Проверить read-only /app, HOME=/tmp,
   размеры tmpfs/логов/кэшей, запуск без runtime-загрузок и поддержку hard links
   в каталоге задачи. Временный ответ создаётся рядом с answer.txt, поэтому граница
   между /tmp tmpfs и /data сама по себе не создаёт cross-device link.
   Dockerfile, Linux lock и контейнерная проверка относятся к A6 и здесь не добавлены.

Также следует согласовать общий/локальный timeout и mode: текущие значения CLI
120 с на запуск и до 30 с на попытку — настройки A5-lite, не подтверждённый
production-профиль соревнования.

**Fake B расположены только в `tests/integration/fakes.py`.** В тестовом каталоге
полностью автономный маршрут запускается так:

```powershell
$env:BUDGET = '100.5'
python run_agent.py --tasks-dir <synthetic-tasks> --budget-unit competition_tokens --components tests.integration.fakes:build --staging-dir <existing-private-temp-parent>
```

Тестовый Solver читает публичный `input.txt` инструментом в staging. Это проверка
связей, не универсальный решатель и не режим сдачи реальных задач. expected нигде
не используется как ответ. `--staging-dir` полезен в ограниченной Windows-среде,
где безопасное открытие предков системного TEMP запрещено; по умолчанию используется
системный временный каталог, небезопасного fallback нет.

`--timeout-s` задаёт дедлайн запуска (по умолчанию 120 с), `--mode` передаёт режим
планировщику. CLI выводит JSON со статусами задач, загрузки, расходом и счётчиками
SDK. Код 0 означает завершённый запуск без LoadIssue, а не решение всех задач;
нерешённые задачи отражены в JSON, их answer.txt отсутствует. Ошибки конфигурации
дают код 2, проблемы безопасной загрузки/LoadIssue — код 1.

**Единицы BUDGET всё ещё не подтверждены организаторами.** `--budget-unit` обязателен:
`competition_tokens` явно преобразует 100.5 в 1005 целых Units, `units_x10` принимает
целые внутренние единицы. Неявного предположения о масштабе нет. A3 strict-mode
не выключен, ProviderProfile.verified не выставлялся. Реальный профиль NeuralDeep
и его интеграционные предположения остаются открытыми по отчёту A3.

## Проверки

Из корня репозитория, после установки `requirements-dev.txt`, `requirements-llm.txt`
и `requirements-sdk-probe.txt`:

```powershell
$env:PYTHONPATH = 'src'
$env:MYPYPATH = 'src'
$env:PYTHONDONTWRITEBYTECODE = '1'
python -m unittest discover -s tests -v
python -m ruff check src tests run_agent.py
python -m ruff format --check src tests run_agent.py
python -m mypy src/nsu_budget_agent/contracts src/nsu_budget_agent/budget src/nsu_budget_agent/llm src/nsu_budget_agent/telemetry src/nsu_budget_agent/task_io src/nsu_budget_agent/planning src/nsu_budget_agent/orchestration/engine.py src/nsu_budget_agent/orchestration/runtime.py src/nsu_budget_agent/orchestration/cli.py src/nsu_budget_agent/submission run_agent.py tests/contracts/typecheck_consumers.py tests/contracts/typecheck_budget.py tests/contracts/typecheck_llm.py tests/contracts/typecheck_planning.py tests/contracts/typecheck_orchestration.py
```

Добавлено 33 теста: 18 сквозных и 15 публикации/политики. Проверяются реальный SDK,
отдельный CLI-процесс, точный ответ, несколько задач, ошибки/отказы/пустой ответ,
лимит шагов, нулевой бюджет для deterministic, недостаток бюджета, учёт usage,
timeout/отмена после отправки и удержание резерва, отложенная повторная попытка,
отмена в момент публикации, старый ответ, повторная публикация, integrity/conflict,
защита файлов, блокирующая неопределённость и отказ файловой системы. После review
добавлены 8 регрессий: изменения исходников и staging при подготовке ответа,
обе очередности координируемого source I/O и commit, точный хеш повторного текста,
изоляция assess, распространение отмены/системных исключений и просроченное открытие.
Fake transport
не использует сеть; audit hook дополнительно запрещает выбранные внешние socket
connect/getaddrinfo-события, **не гарантируя полной сетевой изоляции**.

Локальный результат Windows/Python 3.11: 174 tests, OK (skipped=5), то есть
169 выполнены, 5 ожидаемых платформенных пропусков A4. Ruff lint/format-check
пройдены; Mypy — без ошибок в 34 файлах. YAML разобран установленным PyYAML,
проверены триггер ветки, ubuntu-latest/Python 3.11 и наличие команд A1–A5.

`.github/workflows/a1-ci.yml` расширен: ветка A5, весь discovery A1–A5,
реальный SDK для integration tests, Ruff с entry point, Mypy новых модулей и
Protocol consumers. Существующий SDK smoke сохранён отдельным job. Зависимости
разрешаются для Linux; Windows lock не используется. Timeout job 20 минут,
suite — 5 минут. **Linux/GitHub Actions пока не запускался и не объявляется успешным.**

## Границы A5-lite и риски

- Publisher реализует только answer-only: непустой исходный diff возвращает
  unsupported без изменения исходников и без answer.txt. Транзакции нескольких
  файлов и полноценная сдача coding-изменений требуют следующего этапа A5.
- Выполнение последовательное. max_parallel_tasks — верхняя граница, а не обещание
  параллельности; Ledger/Gateway сохраняют свои ограничения и строгий учёт.
- Изоляция staging и проверки путей не являются OS sandbox для недоверенного
  Python-кода, subprocess или злонамеренных B-компонентов. Доверенные компоненты
  не должны обращаться к task_root/expected напрямую или оставлять фоновые задачи.
  Read-only атрибут сам по себе не доказательство неизменяемости; действуют хеши.
- Открытия используют проверенные handles A4; POSIX записи привязаны к dir_fd,
  Windows — к удерживаемым каталогам и проверке идентичности. Не заявляется
  абсолютная защита от любых внешних гонок/изменения предков привилегированным кодом.
  **Модель угроз commit:** собственные операции в одном процессе соблюдают
  source_operation; B не пишет в оригиналы и не оставляет фоновых писателей.
  Повторный snapshot обнаруживает изменения до его чтения/во время подготовки,
  но процессный guard не блокирует другие процессы или код, обходящий guard.
  Между последним чтением исходников и os.link остаётся окно внешней записи,
  а после commit внешний писатель может изменить исходник/ответ. Обычная проверка
  хеша не устраняет это окно. Для более сильной гарантии нужна отдельно проверенная
  ОС-изоляция или согласованная блокировка всех писателей в B/A6. Остаточная гонка
  не объявляется устранённой, no-clobber answer.txt сохраняется независимо от неё.
- Нужна поддержка hard links целевой ФС; иначе публикация безопасно отказывает.
  fsync каталога best-effort; при сбое удаления временного имени после commit
  ответ уже сохранён, возможен остаточный скрытый временный файл. Журнал в памяти,
  не crash recovery. Внешняя модификация после commit не предотвращается.
- Локальный I/O worker при отмене дожидается завершения: медленная/зависшая ФС
  способна задержать остановку. Убийство процессов и ОС-лимиты инструментов —
  ответственность реального ToolExecutor, не реализованного fake-компонента.
- Семантическая корректность ответов, скрытые тесты, настоящие B и NeuralDeep,
  Linux-файловая семантика и результаты CI остаются отдельной интеграционной проверкой.
