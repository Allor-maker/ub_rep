# Contracts v1 — нормативная спецификация интерфейсов

Версия 1.0, 7 ноября 2026. Это принятые архитектурные решения для реализации на
Python 3.11, а не проект для повторного согласования. Работающих модулей здесь нет.
Общие определения реализует A в `contracts/models.py` (DTO) и `contracts/ports.py`
(Protocol и SolverContext, содержащий LLMSession). ports импортирует models,
models не импортирует ports; это исключает цикл из-за контекста с capability.
B импортирует их, не создавая копий. Следующее изменение границы — отдельный PR
с обновлением этой спецификации и контрактных проверок.

Обозначения источника: **[А]** — выбор архитектора; **[Р]** — локальный регламент;
**[В]** — внешнее неизвестное, изолированное адаптером. SDK-цель — openJiuwen 0.1.19
из регламента; её работоспособность ещё не проверена. Ни одна сигнатура ниже не
выдаётся за API openJiuwen: это собственные порты проекта.

## 1. Общие правила типов и состояния [А]

- DTO — `@dataclass(frozen=True, slots=True, kw_only=True)`; протоколы — `typing.Protocol`.
  Без Pydantic, event bus и иерархии плагинов. `Literal` фиксирует служебные состояния,
  но не категории задач, MIME-типы, ID решателей и инструментов.
- В таблицах поле обязательно, если явно не указано `= ...`. `T | None` без значения
  по умолчанию означает обязательную явную передачу неизвестности. `None` никогда
  не подменяется нулём, пустым ответом или вероятностью успеха 0.
- `Id = str`: непустая строка до 128 символов, без управляющих символов. ID не является
  путём. ID попыток и запросов генерирует A (UUID4 hex); `task_id` приходит из входа.
- `ModelId` — Literal четырёх имён: `qwen3.8-27b`, `qwen3.8-27b-noreason`,
  `qwen3.6-35b-a3b`, `qwen3.6-35b-a3b-noreason` [Р].
- `Units = int`: десятые доли конкурсной единицы, целое >=0; bool вместо int запрещён.
  Полный расход в Units: `input_tokens + 10 * billable_output_tokens` [Р].
- `RelPath = PurePosixPath`: относительный непустой путь внутри `work_root`, без
  `..`, `.`, пустых сегментов, обратного слеша, диска, UNC и NUL. Отдельно разрешён
  `.` только в параметре каталога инструмента. Проверять исходную строку до
  нормализации PurePosixPath. ОС-пути — абсолютные `pathlib.Path`.
- `JsonValue` — JSON: null/bool/int/конечный float/str/list/dict со строковыми ключами;
  `JsonObject = dict[str, JsonValue]`. На границах JSON глубоко копируется, не хранит
  разделяемые изменяемые ссылки. DTO неизменяемы по контракту, включая вложенные JSON.
- Все коллекции DTO — tuple. Время — конечные секунды `loop.time()`; дедлайны абсолютные,
  только внутри одного процесса. UTC-время в логах не используется для таймаутов.
- Неверный DTO/нарушение программистом инварианта — `ContractError(ValueError)`.
  Ожидаемый отказ сервисного порта — `OperationFailed(error: ErrorInfo)`.
  `asyncio.CancelledError` не преобразуется внутри порта в обычную ошибку.
- `ErrorInfo`: `code: str`, `message: str`, `retryable: bool = False`,
  `retry_after_s: float | None = None`. Сообщение <=2048 символов без секретов;
  задержка >=0. Общие коды: `invalid_input`, `unsupported`, `denied`, `timeout`,
  `budget_exhausted`, `quota_exhausted`, `rate_limited`, `provider_error`, `io_error`,
  `invalid_output`, `conflict`, `resource_limit`, `accounting_violation`.
  Дополнительные коды инструментов имеют префикс `tool.`; неизвестный код обрабатывается
  как отказ, а не успех. `retryable` — информация планировщику, не команда повторить.

## 2. Задача и файлы

### TaskSpec — создаёт TaskAdapter A, используют A и доверенный код B

| Поле | Тип / default | Проверка и смысл |
| --- | --- | --- |
| task_id | Id | Уникален в запуске; дубликаты не объединяются |
| instruction | str | Непустая инструкция, без добавленного эталона |
| reward | int | >=0, bool запрещён; политика v1 для входного формата |
| task_root | Path | Канонический каталог задачи внутри входного корня |
| work_root | Path | `workspace/`, если это обычный каталог; иначе task_root |
| artifacts | tuple[ArtifactRef, ...] | Разрешённые исходные файлы, пути уникальны |
| type_hint | str или None = None | Открытая подсказка; неизвестная метка допустима |
| check_hint | str или None = None | Только публичный вид проверки; не эталон |

В v1 нет `metadata: dict` с автоматическим переносом всего манифеста. Из manifest
берутся только перечисленные публичные поля. Новые нужные поля добавляются адаптером
в версионируемый контракт. Корни доступны доверенному коду, но не сериализуются в
промпты и не служат разрешением читать произвольные файлы.

### ArtifactRef — создают загрузчик/инструменты, используют B и публикация A

| Поле | Тип / default | Проверка и смысл |
| --- | --- | --- |
| path | RelPath | Относительно логического work_root, также внутри staging |
| size_bytes | int | >=0, размер фактических байтов |
| sha256 | str | 64 строчных hex-символа, хеш содержимого |
| media_type | str или None = None | Неизвестный формат допустим; не гарантирует парсинг |
| protected | bool = False | True: файл нельзя менять/удалять ни в попытке, ни при публикации |

Загрузчик исключает сырой task.json, эталоны, служебные каталоги и старые ответы.
Известные проверяющие файлы из входного адаптера/инструкции, включая исходные тесты,
помечаются protected. Неизвестное имя теста не угадывается как безопасный для правки:
инструмент запуска тестов дополнительно защищает выбранные существующие тестовые
файлы до запуска и возвращает их в `protected_paths` результата. Защита монотонна.

### Файловая модель [А]

Для каждой попытки A создаёт отдельную staging-копию только разрешённых artifacts
в `/tmp/nsu-budget-agent/<run_id>/<attempt_id>/`. Это `workspace_root` попытки;
исходные task_root/work_root до принятия не изменяются. B работает через инструменты
только в этой копии. Новые обычные файлы допустимы; ссылки, devices и executable
из файлового ввода не запускаются автоматически. Любой путь проверяется повторно
при I/O; symlink в корнях и артефактах отвергается, обход/гонка разрешения пути
не считается решённой одной проверкой строкового префикса.

`answer.txt`, сырой manifest, служебные области и protected-пути недоступны для записи
решателю. Прочитать `expected` через инструмент так же запрещено, как через prompt.
Представление без workspace исключает manifest и эталоны до копирования.
`evaluation/` работает с отдельной эталонной копией вне агентского представления.

Нет неограниченного shell по умолчанию. Исполнение изменённого Python/произвольных
команд допустимо только инструментом с проверенной ОС-изоляцией чтения/записи,
окружения и сети. Если backend недоступен, такой инструмент недоступен в каталоге
и возвращает unsupported при прямом обращении. Отдельный subprocess, cwd и удаление
ключа из env сами по себе не обеспечивают недоступность исходных manifest и других
задач. Backend и его проверка — ответственность B1, запуск под ограничениями — A6;
это реализационная задача, не новый несогласованный порт между A и B.

## 3. Применимость и ограничения попытки

### SolverOffer — создаёт Solver.assess B, читает планировщик A

| Поле | Тип / default | Проверка и смысл |
| --- | --- | --- |
| task_id, solver_id | Id, Id | Задача и зарегистрированный решатель |
| applicability | Literal['yes','no','unknown'] | unknown допускает исследование |
| required_tools | tuple[str, ...] | Уникальные имена; отсутствие обязательного инструмента исключает offer |
| expected_cost_units | Units или None = None | Прогноз, не резерв и не гарантия |
| success_probability | float или None = None | Конечное значение в [0,1] |
| expected_duration_s | float или None = None | Конечное значение >=0 |
| reason | str = '' | Краткое обоснование, обязательно непустое для no |

`assess` синхронный, чистый и дешёвый: только TaskSpec и каталог инструментов, без
LLM, чтения файлов, запуска тестов и обучения. Неизвестные оценки допустимы.
Общий решатель имеет ID `generic` и applicability unknown, если не знает подход;
он не отклоняется лишь по незнакомому type_hint. Registry — явный список, без
динамических импортов из задания. yes у специализации не запрещает планировщику
выбрать generic. no исключает только данный решатель, не объявляет задачу невыполнимой.

### AttemptBudget — создаёт планировщик A, неизменяемый для остальных

| Поле | Тип | Проверка |
| --- | --- | --- |
| max_cost_units | Units | Дополнительный общий лимит этой попытки |
| max_llm_calls | int | >=0; 0 допускает детерминированную попытку |
| max_steps | int | >=1; шаг, возвращающий кандидата, тоже считается |
| deadline_monotonic | float | Конечный, <= дедлайна запуска, в будущем при старте |
| allowed_models | tuple[ModelId, ...] | Уникальны; непусто, если max_llm_calls >0 |
| tool_limits | ToolLimits | Пределы локальных операций |

`ToolLimits` содержит обязательные положительные `timeout_s: float`,
`max_read_bytes: int`, `max_output_bytes: int`, `max_write_bytes: int`,
`max_process_memory_bytes: int`, `max_process_count: int`.
Порядок allowed_models значим: первый — базовый выбор шага; B может выбрать другой
только из этого tuple. Для сравнения одной модели передаётся singleton tuple.
Это явные параметры запуска, не выдуманные лимиты конкурса. На операцию применяется
минимум собственного ограничения и остатка дедлайна. Проверка кандидата тоже
ограничена дедлайном; без завершённой обязательной проверки публикации нет.

## 4. Бюджет и usage

### UsageRecord — создаёт шлюз A, хранит контроллер A

| Поле | Тип / default | Правило |
| --- | --- | --- |
| request_id, attempt_id | Id, Id | Один request_id на одну физическую отправку |
| model | ModelId | Совпадает с lease |
| status | Literal['exact','unknown'] | exact означает полный оплачиваемый расход |
| input_tokens | int или None | >=0; при exact обязательно известно |
| output_tokens | int или None | Полный оплачиваемый выход, включая reasoning |
| reasoning_tokens | int или None = None | Только детализация; при exact не больше output |

`cost_units` — вычисляемое свойство `input + 10 * output` при exact, иначе None.
`input_tokens` включает весь оплачиваемый вход, в том числе кешированный; скидки
по cached_tokens запрещены [Р]. `total_tokens` API не используется как стоимость. Частичные счётчики можно хранить
при unknown, но нельзя считать полным расходом или освобождать по ним резерв.

### BudgetLease — снимок, создаёт и изменяет только контроллер A

| Поле | Тип / default | Правило |
| --- | --- | --- |
| request_id, attempt_id | Id, Id | Уникальная запись контроллера |
| model | ModelId | Из allowed_models попытки |
| input_bound_tokens | int | >=0; учитывает весь сериализованный запрос |
| output_cap_tokens | int | >0; предел всего оплачиваемого выхода |
| state | Literal['reserved','dispatched','uncertain','settled','released'] | Начальное reserved |
| usage | UsageRecord или None = None | exact при settled; unknown при uncertain |

`reserved_units = input_bound_tokens + 10 * output_cap_tokens` вычисляется.
Объект наружу — immutable snapshot; возвращённый снимок не даёт права менять баланс.

### Транзакции контроллера [А]

Контроллер один на процесс, методы синхронные без await, вызываются только из
одного asyncio event loop. Поэтому проверка лимита + резерв + слот вызова выполняются
неделимо. Из threads/subprocess контроллер недоступен; отдельные бюджеты воркеров
не создаются. На одну задачу одновременно допускается только одна попытка.

1. `open_attempt` регистрирует лимиты; повтор с тем же бюджетом идемпотентен,
   с другим — ContractError. `reserve` проверяет глобальные и локальные
   обязательства, дедлайн, allowlist и свободный слот вызова. Повтор reserve того же
   request_id с теми же параметрами возвращает старый lease; с другими — ContractError.
2. `mark_dispatched` происходит до передачи транспорта; после него нельзя считать
   запрос неотправленным лишь из-за timeout/отмены.
3. `release_unsent` разрешён только из reserved: снимает резерв и слот. После dispatch
   нулевой расход подтверждается только exact UsageRecord с нулевыми счётчиками.
4. `record_usage(exact)` заменяет резерв фактическим расходом, состояние settled.
   Слот отправленного вызова остаётся израсходованным даже при нулевом usage.
5. При неизвестном исходе `record_usage(unknown)` переводит dispatched в uncertain;
   полный резерв продолжает удерживаться и глобально, и в попытке.
6. Поздний exact для uncertain учитывается один раз и освобождает разницу.
   Идентичный повтор записи — no-op. Unknown после exact игнорируется. Два разных
   exact для request_id — accounting_violation, остановка новых LLM-вызовов.
7. Если фактический расход выше резерва, он всё равно полностью списывается;
   контроллер блокирует новые запросы и фиксирует accounting_violation. Счётчик
   нельзя подрезать, чтобы скрыть нарушение лимита.
8. `close_attempt` запрещает новые резервы, но не освобождает uncertain и позволяет
   позднюю сверку. Неотправленные резервы освобождаются, отправленные незавершённые
   после cleanup становятся uncertain. Закрытие идемпотентно.
9. `halt(error)` блокирует только новые резервы/отправки, не стирая usage и удержания;
   применяется при окончательной квоте прокси и accounting_violation. Поздний usage
   и локальное сохранение результатов остаются доступны; вызов идемпотентен.

Переходы reserved → dispatched/released, dispatched → uncertain/settled,
uncertain → settled. Повтор того же mark_dispatched или release_unsent идемпотентен,
обратные переходы запрещены. Usage для released либо неизвестного request_id —
ContractError; exact до mark_dispatched также запрещён. Insufficient budget в reserve
возвращается через OperationFailed(budget_exhausted), без отправки и без нового lease.
После halt уже reserved запрос нельзя dispatch: release_unsent освобождает его.
Запросы, отправленные до halt, сохраняют прежние обязательства.

`BudgetSnapshot`: обязательные `limit_units: Units`, `spent_units: Units`,
`held_units: Units`, `calls_used: int`, `blocked: bool`; вычисляемое
`available_units = max(0, limit_units - spent_units - held_units)`.
В global snapshot calls_used — сумма, в attempt snapshot — локальное число занятых
или израсходованных слотов. `held_units` включает reserved/dispatched/uncertain.

Пример: вход 120 и максимум выхода 60 дают резерв 720 Units (72 конкурсных токена).
Usage 120/40 даёт 520 Units: освободить 200, а не списать 520 дополнительно к 720.
Timeout оставляет 720 удержанными до авторитетного уточнения. Второй параллельный
запрос проверяется уже с учётом этих 720.

**[В] Граница гарантии.** `ProviderProfile` A содержит `reasoning_mode:
Literal['included','separate','unverified']`, `input_bound_verified: bool`,
`total_output_cap_verified: bool`; по умолчанию unverified/False/False.
При included completion уже содержит reasoning; при separate счётчики складываются
ровно один раз; при unverified usage считается unknown, а не угадывается по тексту.
Для строгого режима нужны подтверждённые верхняя граница входа и предел полного
выхода. Источник — адаптер прокси/проверенный токенизатор с framing, не chars/4.
В `accounting_mode='strict'` (default) непроверенный профиль запрещает сетевую отправку
с unsupported. `best_effort` разрешён только явно выбранным профилем запуска:
он использует консервативную оценку, но не заявляет гарантию и не подменяет уточнение Q1.
Детерминированные задачи работают и при отключённых LLM-вызовах.

## 5. LLM-порт [А]

Минимальный протокол — текстовый chat, без зависимости от native tools/JSON mode API.
B описывает действия в prompt и проверяет JSON ответа локально. Это одинаковый порт
для общего и специализированных решателей; расширение native tools отложено.

| DTO | Точные поля |
| --- | --- |
| ChatMessage | `role: Literal['system','user','assistant']`, `content: str` |
| LLMRequest | `model: ModelId`, `messages: tuple[ChatMessage, ...]`, `max_output_tokens: int` |
| LLMReply | `request_id: Id`, `text: str`, `finish: Literal['stop','length','other']`, `usage: UsageRecord` |

messages непусты; max_output_tokens >0. Шлюз возвращает обычный текст ответа,
reasoning не добавляется в text и не передаётся как действие инструмента.
Пустой текст допустим для диагностики, но не является корректным действием.
Ответ с length/other нельзя выполнять как частичный JSON или автоматически принимать.
Полные результаты локальных инструментов не включаются в prompt без ограничения размера.

`LLMSession` привязан к attempt_id, step_index и тому же AttemptBudget адаптером A.
Шлюз проверяет индекс шага и использует переданный дедлайн; ledger независимо
контролирует зарегистрированные лимиты. Привязка другого бюджета к тому же attempt_id
— ContractError. Один вызов complete
на шаг, одна физическая HTTP-отправка на complete; SDK/framework retries отключены.
Повтор решения разрешает только планировщик новой попыткой. При ожидаемом отказе
complete бросает OperationFailed после фиксации расхода, при отмене фиксирует
неопределённость отправленного запроса в finally и пробрасывает CancelledError.
Ошибка 429 сама по себе не доказывает бесплатность и не означает всегда конец бюджета.

Ключ и base_url доступны только шлюзу из окружения [Р]. В DTO, prompt, schema,
subprocess env и логах их нет. Успешный текст при неизвестном usage может использоваться,
но резерв не освобождается. Нельзя считать такой запрос бесплатным.

## 6. Инструменты и единственный цикл исполнения

### Данные инструментов — B

| DTO | Точные поля / defaults |
| --- | --- |
| ToolSpec | `name: str`, `description: str`, `input_schema: JsonObject`, `output_schema: JsonObject`, `mutates_files: bool`, `available: bool = True`, `unavailable_reason: str \| None = None` |
| ToolCall | `tool_name: str`, `arguments: JsonObject` |
| ToolContext | `attempt_id: Id`, `workspace_root: Path`, `source_artifacts: tuple[ArtifactRef, ...]`, `protected_paths: tuple[RelPath, ...]`, `deadline_monotonic: float`, `limits: ToolLimits` |
| ToolResult | `call_id: Id`, `tool_name: str`, `status: Literal['ok','error']`, `data: JsonValue = None`, `artifacts: tuple[ArtifactRef, ...] = ()`, `protected_paths: tuple[RelPath, ...] = ()`, `error: ErrorInfo \| None = None`, `duration_s: float = 0.0`, `truncated: bool = False` |

ToolSpec имеет уникальное имя; input/output schemas описывают JSON без удалённых
$ref и callable. При unavailable reason обязателен. ToolResult ok требует error=None;
error требует ErrorInfo и data=None. duration >=0. artifacts — ссылки на результаты
в staging, не разрешение менять исходник. protected_paths добавляются к защищённым
путям A до следующего шага; существующая защита не снимается. source_artifacts — исходный манифест TaskSpec, включая исходные хеши ещё не
распознанных как тесты файлов. Перед первым запуском обнаруженного исходного теста
инструмент сравнивает его с этим хешем: подмена до установки protected не допускается.
Процессы тестов защищены на время самого запуска инструментом, не только после
его результата. Сгенерированные дополнительные тесты живут в scratch и не заменяют
исходные обязательные тесты.
При truncated data соответствует схеме ограниченного вывода, а не случайно обрезанному
JSON; полнота содержимого не утверждается. call_id выделяет A на вызов инструмента.

### Решатель предлагает один шаг, A исполняет цикл

| DTO | Точные поля / defaults |
| --- | --- |
| CandidateAnswer | `text: str`, `evidence_paths: tuple[RelPath, ...] = ()` |
| StopAttempt | `status: Literal['no_progress','unsupported','blocked_by_evidence']`, `reason: str` |
| StepRecord | `step_index: int`, `call: ToolCall`, `result: ToolResult` |
| SolverContext | `task: TaskSpec`, `attempt_id: Id`, `step_index: int`, `budget: AttemptBudget`, `tools: tuple[ToolSpec, ...]`, `llm: LLMSession` |

`SolverStep = ToolCall | CandidateAnswer | StopAttempt`. Candidate text — точный
ответ без автоматического strip/Markdown-обёртки; UTF-8, без NUL, размер ограничен
max_output_bytes. Пустая строка технически допустима, её корректность проверяет
валидация по задаче. evidence_paths должны существовать в разрешённом представлении;
их наличие не доказывает корректность ответа. StopAttempt.reason непустой.

Solver.assess/step не имеют собственного внешнего цикла, subprocess, HTTP-клиента,
доступа к публикации и бюджету. Экземпляры решателей stateless; история передаётся
как tuple StepRecord, начиная с пустой. step_index начинается с 0; один step может
сделать не более одного LLM complete и возвращает ровно одно действие. Внутренние
рассуждения/диалоги не дублируются в скрытом состоянии между задачами.

Текстовый протокол действий generic фиксирован (не API провайдера):
`{"kind":"tool","tool_name":"...","arguments":{...}}`,
`{"kind":"answer","text":"...","evidence_paths":[]}` либо
`{"kind":"stop","status":"no_progress","reason":"..."}`.
Для stop допустимы три статуса StopAttempt. evidence_paths можно опустить (default []).
Неизвестные ключи/тип действия или невалидный JSON — StopAttempt(no_progress), без
исполнения частичных данных. Содержательные schemas arguments берутся из ToolSpec.
Специализированный детерминированный step может вернуть DTO без JSON/LLM.

**Единственный цикл принадлежит AttemptEngine A через JiuwenAdapter.** Он вызывает
step, исполняет ToolCall через B, добавляет StepRecord и переходит к следующему шагу.
Обычная ошибка инструмента попадает в историю для исправления; denied остаётся
ошибкой, а не расширяет полномочия. При Candidate/Stop цикл заканчивается. При лимите
шагов — no_progress, при превышении дедлайна — deadline. B не повторяет этот цикл
внутри generic. Framework не имеет отдельного автономного LLM/tool-loop.

## 7. Кандидат, проверка и решение о принятии

### AttemptResult — создаёт Engine A из действий B, используют validator и planner

| Поле | Тип / default | Правило |
| --- | --- | --- |
| attempt_id, task_id, solver_id | Id, Id, Id | Соответствуют назначенной попытке |
| status | Literal['candidate','no_progress','unsupported','blocked_by_evidence','budget_exhausted','deadline','tool_error','provider_error','cancelled','internal_error'] | candidate ещё не означает решение |
| workspace_root | Path | Отдельная staging-область A, не путь от модели |
| candidate | CandidateAnswer или None = None | Есть только при status=candidate |
| changes | tuple[FileChange, ...] = () | Вычисляет A по фактическому diff staging |
| snapshot_sha256 | str или None = None | Обязателен при candidate |
| steps_used | int = 0 | >=0, не выше max_steps |
| usage_request_ids | tuple[Id, ...] = () | Ссылки на записи контроллера; не отдельный баланс |
| error | ErrorInfo или None = None | Обязателен для tool/provider/internal_error |
| reason | str = '' | Обязателен для любого не-candidate результата |

`FileChange`: `path: RelPath`, `operation: Literal['write','delete']`,
`sha256: str | None = None`; write требует SHA256 нового содержимого, delete — None.
Пути уникальны, protected/reserved-пути запрещены. Удаление только существующего
разрешённого файла. Не-кандидат не имеет changes/snapshot для публикации.

После Candidate A останавливает инструменты, строит diff и замораживает staging.
Snapshot SHA256 — хеш UTF-8 JSON `[candidate.text, manifest]`, где manifest — список
`[path.as_posix(), size_bytes, sha256, protected]` всех разрешённых файлов staging,
отсортированный по path, сериализация `ensure_ascii=False, separators=(',', ':')`.
Защита включает добавленные тестовые пути. Публикация также проверяет diff с исходным
манифестом. Любая правка после проверки инвалидирует snapshot. Validator использует
копию для тестов, которые создают временные файлы; проверяемая версия остаётся неизменной.
Если validator впервые выявил дополнительный исходный тест, он сравнивает его
с исходным ArtifactRef и возвращает blocking failed при изменении/удалении. Это
не переписывает frozen snapshot или protected-флаги кандидата после его создания.
Временные файлы инструмента/валидатора и __pycache__ не включаются в staging:
для них используется отдельный scratch; байткод не записывается рядом с исходниками.

### ValidationResult — создаёт B, не использует expected

Поля: `attempt_id: Id`, `snapshot_sha256: str`,
`checks: tuple[CheckResult, ...]`, `status: Literal['passed','failed','inconclusive']`.
`CheckResult`: `check_id: str`, `status: Literal['passed','failed','inconclusive']`,
`blocking: bool`, `message: str`. check_id уникальны, checks непусты.
Агрегация: есть failed → failed; иначе есть inconclusive → inconclusive; иначе passed.

Обязательные check_id: `format`, `paths`, `protected_files`, `snapshot`, все blocking.
Они проверяют корректный формат результата, допустимые пути/diff, сохранность
защищённых файлов и соответствие snapshot. protected_files сравнивает с исходными
хешами объединение protected-путей TaskSpec, ToolContext и вновь обнаруженных тестов;
исходный protected=False не разрешает правку файла, позднее распознанного как тест. Если требуются доступные
тесты, проверка `visible_tests` также blocking. Отсутствующая семантическая проверка обозначается
отдельным nonblocking inconclusive: её нельзя скрыть за успешной проверкой формата.
Неизвестный check_hint не приводит к догадке `exact_int`. Для unittest корректность
по видимым тестам остаётся локальной; скрытые тесты не объявляются пройденными.

### AcceptanceDecision — создаёт только политика A

Поля: `attempt_id: Id`, `snapshot_sha256: str`,
`status: Literal['accept','reject']`, `reason: str` (непустой).
Порт decide вызывается только для candidate с snapshot; иначе ContractError.
Несовпадение attempt_id/snapshot результата и валидации даёт reject.
Политика v1 определена: accept только при совпадающем snapshot,
отсутствии failed и при passed у всех blocking-проверок. Политика повторно проверяет
наличие четырёх обязательных check_id и blocking=True у них; пропущенная проверка
означает reject, даже если общий status ошибочно передан как passed. Nonblocking inconclusive
не препятствует публикации; это сохраняет возможность ответа без локального оракула.
Ошибка validator создаёт blocking inconclusive, а не разрешение принять результат.
Ни accept, ни passed не является конкурсным баллом.

## 8. Публикация [А], внешний формат [В]

`PublicationPlan`: `answer_path: Path`; путь вычисляет SubmissionLayout A из TaskSpec
и доверенной конфигурации, никогда из текста модели. Формат v1 — один answer.txt.
Default layout проекта — task_root/answer.txt, UTF-8 без BOM и без добавленного
перевода строки. Это выбор по явному пункту регламента, а не подтверждение Q2.
Layout work_root выбирается только конфигурацией. OUT не угадывается: если жюри
подтвердит иной/дополнительный формат, меняется адаптер публикации, не SolverStep.

`SubmissionReceipt`: `task_id: Id`, `attempt_id: Id`,
`status: Literal['saved','already_saved','failed','conflict']`,
`answer_path: Path | None = None`, `answer_sha256: str | None = None`,
`error: ErrorInfo | None = None`. saved/already_saved требуют пути и хеша, error=None;
failed/conflict требуют error. Receipt подтверждает публикацию, не правильность.

Порядок публикации под блокировкой задачи:

0. Сначала проверить журнал публикаций текущего запуска (A хранит receipt и
   принятый snapshot/diff). Для той же попытки с теми же хешами, если сохранённые
   файлы всё ещё соответствуют журналу, вернуть already_saved. Для другого
   кандидата либо изменённых сохранённых файлов — conflict без записи. Эта проверка
   идёт до сравнения оригинала с исходным TaskSpec, поскольку успешная сдача уже
   могла изменить оригинал.
1. Проверить accept, snapshot, protected-файлы и неизменность исходных файлов
   относительно TaskSpec. При конфликте ничего не перезаписывать.
2. Подготовить изменения из staging, резервные копии затронутых исходников и
   временные файлы в той же файловой системе, что и каждый целевой файл.
3. Применить разрешённые изменения к оригинальному work_root; проверить хеши.
4. Записать точные байты CandidateAnswer.text во временный файл, flush/fsync,
   затем атомарно заменить answer_path и при поддержке fsync родительского каталога.
5. Только после сохранения вернуть saved и отметить задачу submitted.

**Ограничение:** набор исходников и answer.txt не образуют одну атомарную транзакцию
ФС. answer.txt публикуется последним как маркер готовности. Сбой до него означает
несданную задачу; при управляемом сбое откатить изменения, а если откат не удался,
заблокировать эту задачу с failed. Сохранность других задач не зависит от отката.

В v1 первая принятая публикация задачи окончательная: новые попытки над submitted
не запускаются. Повтор publish той же попытки/хешей в одном запуске возвращает
already_saved без записи; другой результат — conflict. Существующий на старте
answer.txt — конфликт, а не доказательство успеха; нужен свежий входной набор.
Так не повреждается ранее сохранённый ответ при попытке улучшить его. Возобновление
после перезапуска процесса и изменение уже сданного ответа отложены за v1.

## 9. Порты, async и владение

Сигнатуры ниже — объявления наших Protocol, не реализация и не импорты SDK.
Каждая объявляется методом соответствующего Protocol с обычным `self`;
в краткой записи self опущен. Точные имена Protocol в contracts/ports.py:

| Protocol | Владелец | Методы |
| --- | --- | --- |
| TaskLoader | A | load |
| TaskAdapter | A | load_task |
| SolverRegistry | A | register, all_solvers, get |
| Solver | B | свойство solver_id, assess, step |
| Planner | A | choose |
| BudgetController | A | open_attempt, reserve, mark_dispatched, release_unsent, record_usage, snapshot, close_attempt, halt |
| LLMGateway | A | bind |
| LLMSession | A; вызывает B | complete |
| ToolExecutor | B | catalog, execute |
| AttemptEngine | A; реализация JiuwenAdapter | run_attempt |
| Validator | B | validate |
| AcceptancePolicy | A | decide |
| SubmissionLayout | A | plan |
| Publisher | A | publish |

SolverContext — единственный DTO в ports.py: он зависит от LLMSession. DTO
в models.py содержат только данные и не ссылаются на конкретные сервисы.

```python
# A: загрузка; TaskAdapter выбран доверенным профилем, а не кодом из задачи.
async def load(tasks_dir: Path) -> LoadReport: ...
async def load_task(task_root: Path) -> TaskSpec: ...  # TaskAdapter

# A: реестр (меняется только на старте).
def register(solver: Solver) -> None: ...
def all_solvers() -> tuple[Solver, ...]: ...
def get(solver_id: str) -> Solver: ...

# B: любой решатель, включая generic; solver_id: str — уникальное свойство.
def assess(task: TaskSpec, tools: tuple[ToolSpec, ...]) -> SolverOffer: ...
async def step(ctx: SolverContext, history: tuple[StepRecord, ...]) -> SolverStep: ...

# A: планировщик; чистая функция над снимками, без I/O.
def choose(tasks: tuple[TaskSpec, ...], offers: tuple[SolverOffer, ...],
           progress: tuple[TaskProgress, ...], budget: BudgetSnapshot,
           config: RunConfig, now: float) -> PlanDecision | None: ...

# A: один контроллер, sync-методы только в event loop.
def open_attempt(attempt_id: Id, budget: AttemptBudget) -> None: ...
def reserve(request_id: Id, attempt_id: Id, model: ModelId,
            input_bound_tokens: int, output_cap_tokens: int) -> BudgetLease: ...
def mark_dispatched(request_id: Id) -> BudgetLease: ...
def release_unsent(request_id: Id) -> BudgetLease: ...
def record_usage(usage: UsageRecord) -> BudgetLease: ...
def snapshot(attempt_id: Id | None = None) -> BudgetSnapshot: ...
def close_attempt(attempt_id: Id) -> None: ...
def halt(error: ErrorInfo) -> None: ...

# A: gateway связывает с попыткой/шагом; B видит только LLMSession.complete.
def bind(attempt_id: Id, step_index: int, budget: AttemptBudget) -> LLMSession: ...
async def complete(request: LLMRequest) -> LLMReply: ...

# B: общий исполнитель инструментов, в том числе для validator.
def catalog() -> tuple[ToolSpec, ...]: ...
async def execute(call_id: Id, call: ToolCall, ctx: ToolContext) -> ToolResult: ...

# A: цикл; внутри JiuwenAdapter, а не второй независимо исполняемый агент.
async def run_attempt(task: TaskSpec, solver: Solver,
                      budget: AttemptBudget) -> AttemptResult: ...
# B: локальная валидация только candidate; A: политика принятия.
async def validate(task: TaskSpec, result: AttemptResult,
                   ctx: ToolContext) -> ValidationResult: ...
def decide(result: AttemptResult, validation: ValidationResult) -> AcceptanceDecision: ...

# A: формат и единственный publisher.
def plan(task: TaskSpec) -> PublicationPlan: ...  # SubmissionLayout
async def publish(task: TaskSpec, result: AttemptResult,
                  decision: AcceptanceDecision) -> SubmissionReceipt: ...
```

Дополнительные обязательные DTO этих сигнатур:

| DTO | Поля |
| --- | --- |
| LoadIssue | `task_root: Path`, `error: ErrorInfo` |
| LoadReport | `tasks: tuple[TaskSpec, ...]`, `issues: tuple[LoadIssue, ...]` |
| TaskProgress | `task_id: Id`, `status: Literal['pending','running','submitted','deferred','exhausted']`, `attempted_solvers: tuple[str, ...] = ()`, `last_result: AttemptResult \| None = None`, `not_before_monotonic: float = 0.0` |
| PlanDecision | `task_id: Id`, `solver_id: Id`, `budget: AttemptBudget` |
| RunConfig | `total_budget_units: Units`, `deadline_monotonic: float`, `attempt_limits: AttemptLimits`, `mode: Literal['auto','coverage','portfolio'] = 'auto'`, `max_parallel_tasks: int = 2`, `accounting_mode: Literal['strict','best_effort'] = 'strict'` |
| AttemptLimits | `max_cost_units: Units`, `max_llm_calls: int`, `max_steps: int`, `max_duration_s: float`, `allowed_models: tuple[ModelId, ...]`, `tool_limits: ToolLimits`, `max_attempts_per_task: int = 2` |

AttemptLimits — верхние пределы из доверенного профиля; AttemptBudget не превышает
их и доступный остаток. max_duration >0, max_attempts >=1, parallel в [1,5].
Budget не имеет выдуманного default или неподтверждённого конкурсного env-имени.
Неизвестный бюджет — ошибка конфигурации до отправки LLM. Если внешний бюджет
дан в конкурсных единицах, входной адаптер переводит десятичное значение без binary
float: `floor(10 * external_budget)` Units, консервативно округляя вниз.
RunConfig.mode auto выбирает
coverage при двух корректно загруженных задачах, иначе portfolio; это политика проекта,
которую можно переопределить явным mode при уточнении Q4.

LoadReport изолирует ошибки отдельных задач. Дубликаты ID исключаются все с issues;
невалидный корневой каталог — OperationFailed. Планировщик не назначает running/
submitted/exhausted, учитывает attempted_solvers и остаток; None означает отсутствие
допустимого назначения сейчас, а не математическую невыполнимость. Планировщик
также исключает задачи с now < not_before_monotonic. Оркестратор ждёт завершения
текущей попытки или ближайшего not_before, ограниченного дедлайном запуска, и
переоценивает очередь; отсутствие running не завершает запуск при будущем deferred.
В coverage первая попытка каждой задачи
предшествует повторам; низкая награда не служит причиной отказа. Отсутствие оценок
не исключает offer: v1 использует общий профиль лимитов и детерминированный tie-break
по task_id/solver_id. Пустые/неизвестные оценки не становятся нулевой стоимостью.

TaskProgress изменяет только оркестратор A. pending/deferred → running при назначении;
running → submitted только после saved/already_saved, иначе pending для допустимого
повтора, deferred при временной нехватке ресурсов или exhausted при исчерпанных
попытках/окончательном отказе. Submitted терминален в v1. attempted_solvers хранит
по одному ID на фактически начатую попытку (повторы допустимы), last_result относится
к той же task_id. После временного отказа A выставляет конечный not_before_monotonic >=0, например
время наблюдения отказа + retry_after_s; при повторе параметр сбрасывается в 0.
Неудачный rollback публикации закрывает задачу как exhausted.

Registry требует ровно одного generic, уникальные ID; меняется только до запуска.
get неизвестного ID и некорректный контракт — ContractError. Штатные отказ инструмента
и ошибка JSON модели — ToolResult.error или StopAttempt, а не авария всего прогона.
Engine переводит OperationFailed: budget_exhausted/quota_exhausted → budget_exhausted
(при quota_exhausted также блокирует новые LLM-вызовы), timeout → deadline только
при истёкшем дедлайне, иначе provider_error; неподтверждённый профиль → unsupported,
rate_limited/provider_error → provider_error с сохранённой ErrorInfo. Обычный отказ
execute остаётся ToolResult.error; невосстановимый сбой порта tools → tool_error.
Отказ по resource_limit при создании staging → tool_error, без кандидата.
Неожиданный баг
фиксирует internal_error без ложного кандидата. Accounting violation блокирует
все новые LLM-запросы. Loader/validator/publisher ошибки не маскируют успешным статусом.

### Семантика отказов на каждой границе

| Порт | Ожидаемый отказ |
| --- | --- |
| TaskLoader/TaskAdapter | Ошибка отдельной задачи — LoadIssue в load; load_task бросает OperationFailed. Ошибка корня — OperationFailed |
| Registry | Неизвестный ID/дублирование — ContractError |
| Solver.assess | Нет применимости — no; нет знания — unknown. Не ловить это как exception |
| Solver.step | Штатная остановка — StopAttempt; OperationFailed шлюза пробрасывается Engine; CancelledError всегда вверх |
| Planner | Нет назначения сейчас — None; не создавать искусственную задачу/нулевой бюджет |
| BudgetController | Нехватка лимита/слота/дедлайна — OperationFailed; недопустимые переходы/параметры — ContractError |
| LLMGateway/LLMSession | OperationFailed после учёта отправки/usage, без скрытого повторения |
| ToolExecutor | Неверный аргумент от модели, неподдерживаемость, I/O/timeout — ToolResult.error; CancelledError вверх |
| AttemptEngine | Завершает ошибочную попытку явным статусом AttemptResult; не создаёт ложного кандидата |
| Validator | Возвращает blocking inconclusive при штатном сбое проверки, failed при обнаруженной ошибке кандидата; отмена вверх |
| AcceptancePolicy | Применяет точное правило, возвращает accept/reject; нарушение предусловия candidate — ContractError |
| SubmissionLayout | Недопустимая конфигурация/неизвестный профиль — OperationFailed до записи |
| Publisher | Ожидаемый I/O-сбой — failed, изменение исходных/готовых результатов — conflict; неверный decision/связь DTO — ContractError до записи |

Ни один порт не ловит произвольный BaseException и не превращает его в success.
Если сторонняя реализация validator неожиданно бросила OperationFailed, A также
считает это blocking inconclusive; отсутствие остальных проверок не даёт accept.
Источник истины по расходу — ledger, даже когда результат сетевого порта не получен.

### Исполнение, отмена и зависимости

Весь runtime — один asyncio event loop с ограниченными одновременно активными задачами.
Сеть, процессы и длительный файловый I/O — async-порты. CPU/FS-работа выносится в
контролируемый executor, но он не обращается к бюджету. Чистые assess/choose/decide
и операции книги бюджета — sync. Нельзя делать блокирующий subprocess.run внутри loop.

A применяет `asyncio.timeout_at` к сетевому/инструментальному шагу и валидации;
эффективный timeout — минимум лимитов. CancelledError идёт вверх; engine после
cleanup формирует cancelled (либо deadline при собственном timeout). B останавливает
группу дочерних процессов, не только await; отмена thread-задачи не означает остановку
её I/O. Поэтому запись инструментов остаётся только в staging и завершается до его
заморозки/публикации. После отмены нет новых side effects в оригинальном work_root.
Начатая публикация защищена от обычной task-cancellation до commit/rollback в пределах
дедлайна запуска; SIGKILL всё равно подчиняется ограничению файловой транзакции выше.

`JiuwenAdapter` живёт только в orchestration/, реализует run_attempt и переводит
состояние наших DTO в SDK. Конкретные API и остановка SDK проверяются A1 на 0.1.19;
при несовместимости меняется адаптер, а не интерфейсы B. Импорты SDK не появляются
в contracts/, solvers/, tools/ и validation/.

Граф зависимостей: contracts — только stdlib; компоненты A/B зависят от contracts;
llm использует budget; orchestration собирает registry, planner, gateway, tools,
validator и publisher через порты. Solver не импортирует конкретный gateway,
framework или publisher. validation использует ToolExecutor; publisher не вызывает
solver/validator. evaluation импортирует нужные DTO снаружи, runtime не импортирует
evaluation. Конфигурация и ключи не читаются внутри решателей.

## 10. Минимальный объём и проверка совместимости

Все поля выше входят в v1; поля с default могут не передаваться. Derived properties
не хранятся вторыми источниками истины. Отложены: ML-калибровка/интервалы, native API
functions, дополнительные форматы OUT, динамические плагины, несколько кандидатов
одной попытки, возобновление процесса, перепубликация успешно сданной задачи.

Первый интеграционный сценарий фиксирован: задача с незнакомым type_hint, текстовым
артефактом с новым именем, одним чтением через инструмент и строковым ответом.
Test double LLM выдаёт действие, затем кандидата; реальный openJiuwen исполняет
два шага. Эталон доступен только внешнему тесту. Отдельный вариант использует
work_root=task_root. В этом этапе живой API не требуется.

Обязательные контрактные сценарии для будущих tests/contracts/:

- [ ] C01: DTO валидируются, None отличается от 0, тип bool не проходит как Units.
- [ ] C02: незнакомый type_hint/формат, generic и добавление нового Solver без правки ядра.
- [ ] C03: сырые manifest/expected, symlink, traversal и ключ недоступны агентским инструментам.
- [ ] C04: два конкурентных reserve не расходуют общий остаток дважды.
- [ ] C05: included/separate reasoning, unknown usage, поздний exact, повтор и конфликт exact.
- [ ] C06: отмена до/после dispatch, учёт слотов, закрытие попытки с uncertain.
- [ ] C07: один step/один complete, truncated JSON не исполняется, есть предел шагов.
- [ ] C08: protected-файлы и добавленные тесты неизменны, validator проверяет тот же snapshot.
- [ ] C09: candidate/passed/accept/saved различаются; inconclusive допускается только nonblocking.
- [ ] C10: публикация кода раньше answer, сбой/откат, конфликт и повтор без перезаписи успеха.
- [ ] C11: две задачи получают первые попытки в coverage; остановка сохраняет первую сдачу.
- [ ] C12: run_attempt работает через реальный JiuwenAdapter с тестовыми внешними портами.

Внешние неизвестные остаются только в адаптерах: источник бюджета и usage/лимиты
прокси (Q1/Q5), формат вывода/OUT (Q2), среда/сессии (Q3), официальный режим отбора
(Q4), допустимость экспериментов (Q6), дедлайн и сборка (Q7). Это не причина заново
проектировать наши DTO. Реализационная проверка SDK/изоляции ещё предстоит и не
выдаётся за выполненную.
