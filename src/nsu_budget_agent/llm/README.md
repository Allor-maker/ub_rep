# LLM-шлюз — A

`NeuralDeepGateway` реализует `LLMGateway.bind`; возвращённый объект удовлетворяет
`LLMSession.complete`. Общие контракты не менялись, импорта openJiuwen здесь нет.
Будущий workflow передаёт сессию решателю и закрывает попытку через A2 после cleanup.

## Транспорт и конфигурация

Нужен `httpx==0.28.1` из `requirements-llm.txt`. Gateway создаётся внутри того же
asyncio loop, что и BudgetController, и закрывается через `aclose` или `async with`.
Ключ и адрес читаются только при создании: `LLM_BASE_URL` — HTTPS base URL API,
например путь с `/v1` (к нему добавляется `/chat/completions`), `LLM_API_KEY` — Bearer
credential. Нет default endpoint, загрузки `.env` и импорта настроек внутри решателей.
URL с userinfo, query и fragment отвергается. Эти значения не передаются в DTO/события.

Для отправки используется низкоуровневый `AsyncHTTPTransport(retries=0, trust_env=False)`:
один POST, без redirect/auth-loop, cookies, клиентских retries и HTTPX INFO URL-логов.
Проверка TLS включена; HTTP_PROXY/HTTPS_PROXY не меняют получателя. В тестах транспорт
заменяется `httpx.MockTransport`; никаких реальных сетевых запросов не выполняется.
Включение низкоуровневого HTTP-debug вне шлюза не является безопасной телеметрией.

## Проверенные границы и неизвестность

Конструктор требует `input_bound(request, serialized_body) -> int`: доверенную
синхронную верхнюю оценку входа для выбранной модели, включая chat framing и overhead
провайдера. Функция получает именно отправляемые JSON-байты и исходный LLMRequest.
Gateway не содержит токенизатора и не заменяет верхнюю границу оценкой chars/4.

`profiles` содержит ProviderProfile для каждой используемой модели. В default strict
нужны `input_bound_verified=True`, `total_output_cap_verified=True` и известный
`reasoning_mode`. Отсутствующий профиль равен unverified; complete завершается
`OperationFailed(unsupported)` до резерва и HTTP. Флаги устанавливает доверенный
профиль после отдельной проверки; gateway не объявляет их проверенными сам.
`accounting_mode="best_effort"` включается только явно и не даёт строгой гарантии.

На проводе: model из каждого LLMRequest, текстовые messages, `stream=false`, `n=1`,
выходной cap. `output_token_parameter` явно выбирает `max_tokens` (default) или
`max_completion_tokens`. Само наличие параметра не доказывает ограничение reasoning:
его семантику полного оплачиваемого выхода требуется подтвердить на NeuralDeep.

Usage: весь prompt_tokens, без скидки cached_tokens. Included использует
completion_tokens один раз; separate добавляет reasoning_tokens один раз и требует
этот счётчик, включая явный ноль. Поддержаны completion_tokens_details.reasoning_tokens
и top-level reasoning_tokens; конфликт значений делает usage unknown. Unverified,
отсутствующие/некорректные счётчики, bool вместо int и счётчики error-envelope не
считаются exact. total_tokens не используется. Reasoning-текст не попадает в LLMReply.

Резерв создаётся через A2 до очередей и HTTP. Успешный текст с unknown usage разрешён,
но полный резерв остаётся удержанным. Exact учитывается до проверки структуры ответа:
ошибочный choices не превращает уже известный расход в бесплатный запрос.
Нарушение входной или выходной границы вызывает halt/accounting_violation, даже если
общая стоимость внутри резерва. При превышении всей суммы A2 сначала записывает
полный расход. При unknown сохраняется удержание, а не выдуманная точная сумма.

## Ограничения и ошибки

`bind` проверяет step_index и бюджет попытки, кеширует одну capability на пару
attempt/step. Ровно один вызов complete, включая ошибку до отправки; повторный bind
не выдаёт вторую возможность. Новую попытку планирует внешний orchestrator.
Request ID — новый UUID4 hex; одинаковый ID не используется для повторной отправки.

Один gateway на процесс/аккаунт: semaphore (default 2, максимум 5 HTTP-вызовов) и
равномерный интервал отправки (default 60 RPM, максимум 60, без burst). Очередь тоже
ограничена минимумом timeout_s и дедлайна попытки. Retry-After (секунды или HTTP-date)
увеличивает общий cooldown; уже отправленные запросы могут завершиться. Автоповторов нет.
Другие процессы/клиенты того же аккаунта этим limiter не координируются.

До dispatch timeout/cancellation освобождает резерв и слот. После dispatch любой
неподтверждённый исход, включая ConnectTimeout, сохраняется как unknown; CancelledError
пробрасывается после cleanup. Ответ закрывается в finally, тело ограничено по байтам
(default 1 MiB); gzip/streaming API не запрашиваются. Сетевой ответ с некорректным JSON,
дубликатами ключей, native tools или без одной текстовой choice не исполняется.
Finish length/other возвращается явно: gateway не принимает частичный JSON как действие.

429 по умолчанию rate_limited, retryable — только подсказка внешнему планировщику.
Коды error.code `insufficient_quota`, `quota_exhausted`, `budget_exhausted` вызывают
quota_exhausted и глобальный halt; 401/403 — denied/halt. 5xx/408 — retryable
provider_error; остальные HTTP-отказы — provider_error без retry. Тела ошибок и сырые
исключения транспорта не включаются в ErrorInfo. Поля безопасных LLMEvent: локальные
ID, model, phase, usage_status/error_code. Сбой обычного event sink не влияет на учёт.

Все гарантии лимитера относятся к локальным HTTP-операциям. После timeout сервер
может продолжать генерацию; его реальное число активных заданий и quota/cancellation
семантику нужно проверить отдельно. Поздняя сверка unknown возможна только по
авторитетным данным через A2, не по догадке из кода HTTP-ошибки.

Примеры настройки, проверки и открытые предположения — [отчёт A3](../../../docs/a3-report.md).
