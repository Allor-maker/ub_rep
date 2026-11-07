# A3 — NeuralDeep LLM Gateway

Работа в `feature/a3-llm-gateway`, база `42504c6`. Реализован только A3: общие
Contracts v1, A2 Budget Ledger и компоненты B не менялись. A4–A7 не реализованы.
Все новые HTTP-проверки используют MockTransport, реальных NeuralDeep-вызовов нет.

## Реализация

- `llm/gateway.py`: NeuralDeepGateway, bind с бюджетом/индексом шага и одноразовый
  LLMSession, HTTPS OpenAI-compatible chat completions по LLM_BASE_URL/LLM_API_KEY.
  Все четыре модели разрешены на уровне DTO; конкретный вызов также ограничен
  allowed_models попытки. Нет фиксированного выбора модели внутри решателя.
- `llm/usage.py`: included/separate/unverified, partial/unknown counters и точный
  учёт полного оплачиваемого выхода без двойного reasoning и скидки кешированного входа.
- `telemetry/llm.py`: безопасный lifecycle event без ключей, URL, prompt, response body
  и reasoning-текста. Gateway не логирует сырые HTTP-исключения и тела ошибок.
- `requirements-llm.txt`: один прямой runtime dependency `httpx==0.28.1`; его
  транзитивные зависимости разрешаются пакетным менеджером. OpenAI SDK не добавлен,
  openJiuwen остаётся только в orchestration. Финальный Linux runtime-lock — A6.
- Mock HTTP тесты в `tests/unit/test_llm_gateway.py`, структурная проверка unchanged
  LLMGateway/LLMSession — `tests/contracts/typecheck_llm.py`.

Используется low-level HTTPX transport с retries=0 и trust_env=False, без AsyncClient
redirect/auth/retry циклов. Реальные SDK-интерфейсы здесь не появляются. Будущий
JiuwenAdapter может использовать существующий Protocol и выдавать LLMSession в SolverContext.

## Конфигурация и strict

Gateway создаётся внутри event loop, получает общий BudgetController, доверенную
функцию input_bound и profiles по моделям. Endpoint: HTTPS base URL API плюс
`/chat/completions`, key: Bearer header. Нет встроенного endpoint, чтения `.env`,
запуска subprocess, скачивания токенизатора или моделей. Успешный import не отправляет HTTP.

Пример сборки (параметры предоставляет доверенный вызывающий код, не задача):

```python
from nsu_budget_agent.llm import NeuralDeepGateway

async with NeuralDeepGateway(
    budget=ledger,
    input_bound=verified_bound,  # (LLMRequest, точные JSON-байты) -> int, с framing/overhead
    profiles=verified_profiles,
) as gateway:
    llm = gateway.bind(attempt_id, step_index, attempt_budget)
    reply = await llm.complete(request)
```

Strict — default. Для выбранной модели нужны проверенные input/output bounds и
известный reasoning_mode. По умолчанию профили unverified; отправка запрещена с
unsupported. В тестах флаги verified относятся **только к mock-провайдеру**, не к
реальному NeuralDeep. Встроенного доказанного токенизатора/верхней оценки нет.
Для исследования разрешён только явный best_effort с предоставленной оценкой;
это не заявляется как гарантия жёсткого лимита. Все вызовы и там проходят через A2.

`max_tokens` — default output parameter; `max_completion_tokens` выбирается явно
профилем создания gateway. Ни один из них не объявлен подтверждённым NeuralDeep cap
полного выхода. Никакие флаги ProviderProfile не устанавливаются автоматически.

## Учёт и lifecycle

Gateway резервирует весь bound входа и max_output_tokens до ожидания semaphore/rate
slot. Mark dispatched выполняется до передачи транспорта. Отмена/timeout в очереди
освобождает unsent резерв/слот; после dispatch неизвестный исход сохраняет весь резерв.
ConnectTimeout тоже считается неопределённым исходом отправки. Скрытых retry нет.
Повтор complete, в том числе после ошибки до отправки и через повторный bind,
отвергается. Ключи разных шагов могут выполнять новые вызовы в пределах попытки;
решение повторить ошибочную попытку принадлежит будущему оркестратору, не gateway.

Все reported prompt_tokens оплачиваются. Included completion содержит reasoning
и списывается один раз; separate требует detail/top-level reasoning и складывает
его один раз. Unverified/нет usage/невалидные счётчики дают unknown, не бесплатный
вызов. Счётчики error-envelope консервативно не объявляются exact. Успешный текст
при unknown может вернуться потребителю; удержание в A2 не освобождается.

Expense учитывается даже при неправильном choices. Ответ length/other не считается
исполняемым действием — gateway только возвращает текст/finish. Строки reasoning
отдельного поля message не добавляются в text. Проверка generic JSON относится к B.

Закрыт локальный риск review A2: при обнаруженном `output_tokens > output_cap_tokens`
или input выше bound gateway останавливает новые LLM-вызовы, даже если суммарный
расход внутри резерва. Нормализованный usage сначала передаётся ledger. Известный
полный расход сохраняется, частичный остаётся unknown с полным удержанием.
Если стоимость выше резерва, A2 сохраняет её целиком до accounting_violation.

Default ограничения: 2 HTTP-вызова одновременно, максимум 5; до 60 RPM с равномерным
интервалом (без burst). Retry-After увеличивает общий monotonic cooldown. Ошибка 429
не запускает retry и сама по себе не освобождает резерв; окончательная квота/401/403
вызывают halt, в том числе до dispatch уже ожидающих запросов. Новые данные usage
для уже отправленных вызовов по-прежнему учитываются. Timeout 30 секунд по умолчанию
включает очередь, ограничивается дедлайном попытки; ответ ограничен 1 MiB.

Гарантии относятся к одному gateway на процесс. Несколько клиентов одного аккаунта
не координируются автоматически. После timeout модель на стороне сервера может
продолжить генерацию: локальный semaphore не доказывает число remote inference jobs.
Транспорт освобождает локальную операцию, но бюджет сохраняет неизвестное обязательство.
Gateway не закрывает попытку и не публикует ответы — эти действия принадлежат A5.

## Проверки

Из корня проекта, Python 3.11, PowerShell (HTTPX уже доступен в локальной `.venv-a1`):

```powershell
# Для нового окружения: python -m venv .venv-a1
.\.venv-a1\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-llm.txt
$env:PYTHONPATH = 'src'
$env:MYPYPATH = 'src'
.\.venv-a1\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-a1\Scripts\python.exe -m ruff check src tests
.\.venv-a1\Scripts\python.exe -m ruff format --check src tests
.\.venv-a1\Scripts\python.exe -m mypy src/nsu_budget_agent/contracts src/nsu_budget_agent/budget src/nsu_budget_agent/llm src/nsu_budget_agent/telemetry tests/contracts/typecheck_consumers.py tests/contracts/typecheck_budget.py tests/contracts/typecheck_llm.py
```

Локально: **90 тестов успешно** (55 A1/A2 и 35 A3), Ruff lint/format и Mypy успешны.
Новые пакеты локально не устанавливались. Полный suite теперь требует HTTPX;
сам C01 продолжает проверять subprocess import contracts с `-S`, без SDK и сети.

Покрыты четыре модели/точные wire bytes, strict/best_effort, нормализация usage,
cached/total counters, отсутствующий usage, malformed/oversized JSON и choices,
length/other, HTTP 400/5xx/429/401/quota/redirect, Retry-After, ошибки транспорта,
таймаут/отмена до и после dispatch/при чтении тела, closure HTTP stream, repeat bind/
complete и конкурентные вызовы одного шага. Проверены конкурентный бюджет, semaphore,
rate spacing, queued request после halt, individual cap violation внутри общей суммы,
неисправный event sink и отсутствие секретов в собственных events/ErrorInfo.
Real HTTP handle запрещён patch-guard в тестах, остальные ответы — MockTransport.

CI `.github/workflows/a1-ci.yml` расширен на ветку A3: Linux/Python 3.11, полный suite,
HTTPX + dev dependencies и Mypy нового модуля. Отдельный SDK smoke A1 сохранён.
Windows lock не используется. Проверка YAML/команд локальная; GitHub Actions A3
не запускался, Linux-результат не заявляется. Commit/push/merge не выполнялись.

## Исправления после ревью A3 и подготовка A4/A5

- После получения заголовков HTTP 429 общий cooldown устанавливается сразу, до ожидания тела.
  Ожидающий rate limiter повторно проверяет его после сна; уже отправленные вызовы не отзываются.
  Timeout или отмена чтения тела не отменяет cooldown и сохраняет резерв отправленного вызова.
- HTTP-запрос собирается до `mark_dispatched`. Локальная ошибка URL возвращается как
  `OperationFailed(invalid_input)`, а неотправленный резерв освобождается. После передачи
  транспорту неизвестный расход по-прежнему удерживается. Тесты используют MockTransport.
- До подтверждения профиля NeuralDeep A4/A5 могут внедрять тестовую реализацию `LLMGateway` и
  `LLMSession` из Contracts v1 с заранее заданными ответами и ошибками, без сети и ключей.
  Для интеграционных тестов самого A3 можно применять `NeuralDeepGateway` с MockTransport,
  фиктивным ключом и профилем, подтверждённым **только для mock**. Такой профиль и тестовый
  `input_bound` нельзя использовать как свидетельство для реального провайдера; реализация
  A4/A5 в этот этап не входит. Strict по умолчанию остаётся включённым.

До реального strict-запуска необходимо подтвердить для каждой модели верхнюю границу входа
с учётом токенизатора, framing и overhead, ограничение полного выхода включая reasoning,
а также семантику usage и reasoning. Небольшой smoke не доказывает эти границы для всех входов;
нужны надёжные сведения провайдера и проверяемый адаптер. Остальные допущения перечислены ниже.

## Что проверить отдельным разрешённым NeuralDeep smoke

1. Реальные base URL/путь, Bearer auth, TLS и доступность всех четырёх ModelId.
2. Поддержку выбранного max_tokens/max_completion_tokens и cap **полного** выхода,
   включая reasoning; отсутствие скрытой генерации сверх cap и скрытых повторов прокси.
3. Полноту prompt_tokens (framing/скрытые добавления/кеш), completion/reasoning поля,
   included или separate, полноту оплачиваемого usage и поведение error-envelope.
4. Проверенный tokenizer/framing и верхнюю границу входа для каждой модели; маленький
   smoke сам по себе не доказывает bound для произвольного prompt.
5. Реальные quota/error codes, Retry-After и 60 RPM/параллельность, в том числе
   другие клиенты аккаунта и продолжение/списание после timeout/cancellation.
6. Возможность авторитетной поздней сверки неизвестного расхода. Без неё удержание
   остаётся до конца процесса и не объявляется бесплатным.

До этого strict остаётся fail-closed с unverified профилем. Реальные вызовы требуют
отдельного разрешения, явно заданного тестового бюджета и фиксации наблюдаемого расхода.
