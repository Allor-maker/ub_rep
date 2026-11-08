# A7.1 — Decision Router / Baseline v2: исправления review

Ветка `feature/a71-decision-router`. Исправляются восемь воспроизведённых
дефектов независимого review. Новые конкурсные стратегии, coding pipeline,
model router и Portfolio Optimizer не добавлялись.

**Verdict: APPROVED для локальной приёмки исправлений всех восьми дефектов.** Production LLM
по-прежнему требует trusted compatibility adapter; готового адаптера NeuralDeep
нет. Mock-профили и успешные smoke не включают production API.

## Архитектура и совместимость

A4 загружает весь портфель и оценивает все offers до первого workflow. Сохраняются
приоритет полного бесплатного решения, reward-aware ранжирование, глобальный A2
Ledger, A3 strict Gateway и единственная публикация через A5.

DecisionStore и контроллер ролей находятся в `orchestration/decision.py`.
AgentRunner создаёт собственные store/controller/validator/planner, даже если
входные Components разделяются несколькими runners. Каждый run очищает маршруты,
историю результатов и счётчики workflows. Возвращённые traces/results — snapshots.
Повторный run с тем же task_id и новой инструкцией не использует старый маршрут.
Одновременные run одного runner отклоняются ContractError до await. Разные runners
имеют отдельные состояния; общий Ledger/Gateway сохраняет глобальный учёт.

Публичные FreeDecisionSolver и DecisionSolver stateless: хранят только startup
configuration. Обычный v1 consumer получает ToolCall/CandidateAnswer/StopAttempt.
StopAttempt всегда завершает попытку. ContinueDecision удалён полностью: нет ни
subclass, ни специального толкования StopAttempt. Внутренний контроллер A5 может
вернуть None своему JiuwenAdapter для перехода к следующей роли. Это внутренний
результат оркестрации, а не SolverStep; переход расходует SDK-шаг. Каждый шаг
содержит максимум один complete. Публичный v1 generic сохраняет проверяемый
одноролевой fallback; расширенная последовательность Solver/Reviewer выполняется
только внутри A5, с состоянием и продолжением, принадлежащими AttemptEngine.

Contracts v1 (`contracts/models.py`, `contracts/ports.py`, `docs/contracts.md`),
A2/A3 strict и файловые правила A5 не менялись. Новых runtime-зависимостей нет.
Физическая отправка и резервирование остаются исключительно внутри A3/A2.

## Состояния и переходы

| Из | В | Условие |
| --- | --- | --- |
| FREE_ATTEMPT | TOOL_EXECUTION / UNSOLVED / SKIP | Полный/частичный план, отсутствие плана, отказ по источникам/coding |
| TOOL_EXECUTION | UNCERTAIN / LLM_REVIEW / VALIDATION / SKIP | Реальный частичный кандидат, новый LLM-план, полный разбор, ошибка |
| UNCERTAIN | LLM_REVIEW / SKIP | Реальный кандидат и весь неразобранный остаток |
| UNSOLVED | LLM_SOLVE / SKIP | Незнакомая инструкция допущена к bounded Solver |
| LLM_REVIEW | VALIDATION / LLM_SOLVE / SKIP | accept, новая обратная связь после reject/uncertain, отказ |
| LLM_SOLVE | TOOL_EXECUTION / LLM_REVIEW / VALIDATION / SKIP | tool_plan, direct_answer с review, проверяемый ответ, decline/ошибка |
| VALIDATION | CERTIFIED / PUBLISH / SKIP | Независимый разбор и вычисление, review-policy, отказ |
| CERTIFIED | PUBLISH / SKIP | Результат публикации через A5 |
| PUBLISH / SKIP | — | Терминальные состояния |

Если первичный резерв недоступен, задача сохраняет UNSOLVED/UNCERTAIN и
TaskProgress.deferred без платной попытки. A4 снова рассматривает её при каждом
изменении портфельного/бюджетного снимка. Если после реальных ответов вырос резерв
следующего review, отказ по бюджету также возвращает deferred с текущим маршрутом,
не публикует кандидат и не вызывает повторный HTTP. Автоматического возобновления
уже начатого платного маршрута в новом attempt нет: дальнейшее решение принадлежит
вызывающему коду. Следующий run начинает чистое состояние и сохраняет учёт Ledger.
Фонового ожидания позднего exact usage этот последовательный A5 driver не реализует.

## Статус восьми дефектов

Для каждого исходного дефекта сначала добавлена воспроизводящая регрессия,
зафиксирован красный результат и затем внесено исправление.

| № | Статус | Исправление и доказательство |
| --- | --- | --- |
| 1 | FIXED | Закрытая грамматика сущностей отвергает `Alice, not Bob,`, `Alice or Bob`, `Alice from London`, `Alice Rather Than Bob`. Дополнительный независимый разбор привязывает всю существенную инструкцию к плану. Ни один из четырёх отвлекающих профилей не получил CERTIFIED или публикацию в регрессии. |
| 2 | FIXED | Store сохраняет immutable отвергнутые Proposal, включая ответ, evidence и ToolCall. Тот же direct_answer после reject блокируется до второго review; одинаковые ответ+план также не ревьюятся заново. Для прежнего ответа допустим только изменённый исполняемый план с независимой проверкой и новым review. Эксплойт 30 → reject → direct 30 → accept завершён без публикации, 2 HTTP вместо 3. |
| 3 | FIXED | Введена явная selected_sources allowlist: точные ссылки на paths/directories и источники закрытого Markdown-вопроса. Остальные artifacts остаются только в исходном manifest A5, без содержимого и provenance в prompt. Credentials/secrets/password/token/key paths запрещены; известные secret assignments и private-key markers в выбранном тексте блокируются до HTTP. `credentials.txt` в арифметике не появляется ни в одном prompt, арифметика решается; явный запрос credentials и api_key в обычном sales.csv отклоняются до HTTP. |
| 4 | FIXED | Для CERTIFIED нужен отдельный instruction binding и независимый Decimal/AST, CSV DictReader или Markdown parser. Источники читаются через разрешённый executor и привязываются к size/hash входа. Сумма 999, подменённая одновременно в решении и повторном вызове executor, отвергается. Аналогично отвергаются подменённые math/Markdown. Если независимая проверка невозможна, сертифицированной публикации нет. |
| 5 | FIXED | После бесплатного сбора нужных источников planner получает стоимость конкретного минимального live reservation из trusted adapter с тем же wire JSON, что A3. Доля не может быть ниже необходимого резерва; allocation допускает ограниченную цепочку ролей и ограничен общим/локальным бюджетом. Недостаток доступного резерва даёт defer без платной попытки. Rewards 90/50/1 при 24000 Units решаются в таком порядке. A4/A2 stress 100 задач с резервом 13312 сохраняет все 100 и верхние 25. |
| 6 | FIXED | Store принадлежит runner/run. Повторный run с изменённой инструкцией даёт 6 вместо старых 5 и отдельный результат одной попытки. Overlapping run одного runner явно запрещён. Два параллельных runners с общими Components и одинаковым task_id независимо публикуют 5 и 6. |
| 7 | FIXED | Preflight проверяет минимум вызовов и SDK-шагов, модель и доступность минимального резерва до HTTP. Незнакомая задача, требующая Solver+Reviewer, при max_llm_calls=1 не отправляет запросов и тратит 0 Units. При неизвестном будущему тексту ответе последующие реальные bounds вновь проверяются A3/A2. |
| 8 | FIXED | ContinueDecision/StopAttempt-subclass удалён, mutable routing перенесён в A5. Публичные solvers не имеют store. Обычный v1 consumer получает настоящий terminal StopAttempt; повторный вызов с теми же ctx/history даёт тот же результат без скрытого продолжения. Путь review → reject → solve → tool → review → Validator → A5 остаётся реальным SDK workflow. |

Дополнительно сохранён бесплатный охват: четыре Markdown-файла в полностью
распознанной задаче не подпадают под лимит трёх источников LLM. Они решаются и
независимо сертифицируются без HTTP. Платные маршруты готовят ограниченные
источники на бесплатном проходе, чтобы planner мог оценить конкретный reservation
без чтения файлов или отправки HTTP внутри choose/assess.

В ходе полной регрессии обнаружена и исправлена регрессия сброса publisher:
первый run сохраняет внедрённый publisher, повторный получает новый receipt cache.
Исходный тест отмены публикации сохраняет committed/submitted и очищает staging.

## Политика допуска и бюджет

CERTIFIED требует полного связывания инструкции и независимого вычисления.
Программная проверка знакомых A3.5/A3.6 вопросов сохраняется. Для неизвестной
семантики обязательны отдельный typed Reviewer accept без missing_conditions,
exact usage и все обязательные format/paths/protected_files/snapshot checks A5.
Для tool_plan дополнительно независимо проверяется предложенное вычисление.
Semantics при review-only остаётся inconclusive/nonblocking; CERTIFIED нет.
Direct answer ограничен 512 UTF-8 байтами без управляющих символов и использует
только evidence из выбранных источников. При наличии выбранных источников
пустое evidence запрещено. Отвергнутые кандидаты нельзя возвращать как голый
повторный direct_answer.

В одной задаче максимум 3 отправки: initial review, один solve и review нового
плана; для неизвестной задачи обычно solve+review. Одинаковый fingerprint без
новой информации запрещён. Ошибки 429/500/timeout и unknown usage не повторяются.
Unknown/timeout после dispatch сохраняют весь held. Все I/O и проверки ограничены
существующим deadline, а публикация доступна исключительно A5.

Equal-share остаётся простым baseline, но не жёстким потолком ниже reservation.
При отсутствии controller/reservation capability используется исходная аллокация
A4: неизвестная граница не подменяется искусственно маленькой долей. Это также
проверено для самостоятельного DecisionPortfolioPlanner без production composition.
Резерв освобождается после exact usage, поэтому следующая физическая отправка
использует фактический остаток. Allocation учитывает ограниченное число ролей;
он не выдаётся за измеренную стоимость, вероятность или marginal reward.
При недостатке текущего live reservation задача остаётся deferred. Неизвестные
оценки SolverOffer по-прежнему None.

## Проверки и измерения

Новые регрессии:

- `tests/integration/test_a71_review_fixes.py`: исходные эксплойты 1–8, source
  allowlist/secret content, повторные и параллельные runs, defer и сохранение
  бесплатного охвата. Настоящие openJiuwen, A2/A3 MockTransport, tools, Validator, A5.
- `tests/unit/test_a71_allocation.py`: 100 задач, глобальный A4/A2, 200 reservations
  с exact settlement, reward-порядок 100…1, ни одна из верхних 25 не теряется.
- `tests/unit/test_a71_certification.py`: независимая привязка всех существующих
  форм закрытой грамматики, отрицательные floor/modulo, Decimal, power, stock split,
  CSV conditions/schema и Markdown relations.
- Прежние `test_a71_router.py` и `test_a71_decision.py`: FSM, role contracts,
  limits, unknown/held, реальный timeout и сравнение восьми сценариев.

До исправлений адресные новые тесты воспроизвели исходные ошибки. После этапа 1
прошли 23 адресные проверки. После этапа 2 — 41 адресная проверка; дополнительные
secret-content, сохранение большого бесплатного Markdown и cancellation/re-run
проверялись отдельно до включения в полный discovery.

### Одинаковые восемь mock-сценариев

| Композиция | Правильные / 8 | Ошибочные публикации | Безопасные отказы | HTTP | Mock prompt | Mock completion | Units |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B-minimal | 1 | 0 | 7 | 0 | 0 | 0 | 0 |
| A3.6 | 2 | 0 | 6 | 1 | 120 | 48 | 600 |
| A7.1 до review fixes | 6 | 0 | 2 | 11 | 1320 | 528 | 6600 |
| A7.1 после review fixes | 6 | 0 | 2 | 11 | 1320 | 528 | 6600 |

Охват этого набора не снизился. Новые adversarial-регрессии проверяют ошибки,
которых в этих восьми задачах не было; исходные 0 ошибочных публикаций не были
доказательством корректности всех маршрутов.

Каждый успешный scripted HTTP сообщает prompt=120 и completion=48, то есть
120 + 10×48 = 600 Units. Это заданный mock usage, а не измеренная токенизация
или точность NeuralDeep. Cap=512, input bound=4096, live reserve=9216 Units.
В успешном сравнении held=0; unknown/timeout/429/500 удерживают 9216.
Эталоны находятся только в тестовом сравнении, вне TaskSpec, staging и prompt.
Исходные expected/gold не читаются, настоящих API-вызовов нет.

Portfolio-регрессии:

- 3 задачи, rewards 90/50/1, бюджет 24000, reserve 9216: 3 правильных публикации
  в reward-порядке, 6 HTTP, 3600 spent, 0 held. Раньше reward 90 исчерпывался до HTTP.
- Review/replan плюс другая задача, бюджет 20000: теперь публикуются обе,
  5 HTTP, 3000 spent, 0 held. Раньше искусственная доля прерывала третий review.
- 100 задач, budget 1000000, reserve 8192+10×512=13312: чистый A4/A2 stress,
  200 резервов/точных usage, порядок reward 100…1, spent=120000, held=0.
  Это не 100 реальных inference-запросов и не обещание параллельного LLM runtime.
- Последующий review требует 30000 Units вместо первоначальных 9216: high остаётся
  deferred после 2 HTTP, low решается за следующие 2 HTTP; 2400 spent, 0 held,
  непроверенный кандидат high не публикуется, автоматического retry нет.

### Итоговые команды

```powershell
$env:PYTHONPATH = 'src'
$env:MYPYPATH = 'src'
.\.venv-a1\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-a1\Scripts\python.exe -m ruff check src tests run_agent.py
.\.venv-a1\Scripts\python.exe -m ruff format --check src tests run_agent.py
# Полная команда Mypy из CI, дополнительно orchestration/decision.py.
.\.venv-a1\Scripts\python.exe tests/docker/smoke.py --mode host
.\.venv-a1\Scripts\python.exe tests/docker/smoke.py --mode host --source C:\VSCode\openJiuwen\nsu_benchmark_smoke_12_tasks\tasks
```

Ruff lint/format прошли; Mypy — без ошибок в 58 файлах. Host smoke baseline:
3/6 переносимых задач, 7 workflows; публичный smoke — 6/12, 12 workflows.
BUDGET=spent=held=0; нерешённые без answer.txt, исходные наборы не менялись.
Docker/Podman в PATH отсутствуют; Linux/контейнер и GitHub Actions не заявляются
проверенными. Последний полный unittest после всех исправлений: **305 тестов за
138,105 с, OK (skipped=6)** — 299 выполнены, 6 платформенных пропусков. Все 44
проверки A7.1 прошли, включая 19 добавленных при исправлении review. Ruff lint и
format-check прошли (88 файлов), Mypy — без ошибок в 58 файлах; git diff --check
прошёл. Исходные эксплойты повторно проверены также прежними временными скриптами:
999 и отклонённые 30 не публикуются, отрицание не получает CERTIFIED, credentials
отсутствуют в prompt, самостоятельный 100-task planner больше не пропускает top 25.

Журналы стадий и финального прогона находятся в `%TEMP%/a71-stage1-before.log`,
`a71-stage1-after.log`, `a71-stage2-before.log`, `a71-stage2-final.log`,
`a71-fixed-full.log`, `a71-fixed-verified.log`, `a71-fixed-smoke.log`, `a71-fixed-public-smoke.log`.

## Изменённые файлы

- `orchestration/decision.py`: A5-owned store, FSM, role requests/preflight.
- `orchestration/engine.py`, `runtime.py`: внутреннее продолжение, run isolation,
  allocation/defer и неизменная цепочка validation/publication.
- `solvers/decision.py`, `sources.py`, `plans.py`, `composition.py`: stateless v1
  capabilities, источники/секреты, закрытые entity и trusted reservation preview.
- `planning/decision.py`: reservation floor, reward-порядок и deferred offers.
- `validation/decision.py`, `certification.py`, `instruction.py`: независимое
  вычисление/привязка, provenance и ограниченный review-policy.
- Тесты выше, `docs/a71-report.md`, README/tests README. Ранее добавленные CLI traces
  и trigger ветки в существующем CI сохраняются.

## Оставшиеся ограничения

- Неизвестная семантика, принятая Reviewer, остаётся вероятностной. Solver/Reviewer
  могут согласованно ошибиться; прямые ответы без verifier не сертифицируются.
- Secret-name/content policy консервативна и может отказать допустимому файлу.
  Неизвестную секретную строку без распознаваемого имени/формата невозможно
  гарантированно обнаружить. Входной набор должен содержать только публичные данные.
- Для полного local certificate разделён существенный разбор и вычисление;
  общие policy helpers для явно разрешённых execution/output suffixes и формы
  простого имени сохранены; операция/условия разбираются отдельно от plan().
  Новая грамматика требует расширения независимой проверки, иначе безопасный отказ.
- Платные данные ограничены 3 текстовыми источниками и 16 KiB. Ограничение не
  распространяется автоматически на полностью проверяемый бесплатный маршрут.
- Новые ответы модели могут увеличить следующий input bound. A2 остаётся
  окончательным контроллером остатка и может остановить цепочку после новой информации.
  Предварительная оценка не подменяет резерв и не гарантирует будущий usage.
- A5 выполняет задачи последовательно. Overlapping run одного runner запрещён;
  тест изоляции разных runners не делает эту систему параллельным scheduler.
- При отсутствии других действий нет фонового ожидания позднего exact usage;
  deferred задачи возвращаются вызывающему коду. Сохраняются ограничения файловых
  гонок/платформ A5. Trusted adapter NeuralDeep и Linux CI остаются внешними блокерами.

Commit, push, merge и настоящие NeuralDeep-вызовы не выполнялись.
