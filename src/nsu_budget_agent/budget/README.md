# Бюджет — A

`BudgetLedger` реализует неизменённый Protocol `BudgetController` из Contracts v1.
Модуль использует только stdlib и общие DTO; сеть, SDK и решатели не импортируются.

## Единицы и конфигурация

Внутри только целые `Units` (десятые конкурсной единицы):
`cost_x10 = input_tokens + 10 * output_tokens`. Полный `output_tokens` уже включает
reasoning; ledger не прибавляет `reasoning_tokens` повторно и не угадывает usage.

`budget_from_env(unit=...)` читает `BUDGET` только по явному вызову:

- `unit="competition_tokens"`: `floor(10 * Decimal(BUDGET))`;
- `unit="units_x10"`: значение уже в Units, требуется целое число.

Единицы обязательны, default и догадки по величине отсутствуют. Например,
`BUDGET=72` означает 720 или 72 Units в зависимости от выбранного режима.
Отсутствующее/нечисловое/отрицательное/не конечное значение — `ContractError`.
Binary float не принимается. Семантика `BUDGET` у организаторов остаётся Q1;
наличие этого opt-in помощника её не подтверждает.

## Использование

Создавайте ledger и вызывайте все его методы внутри **одного running event loop**.
Другие loop и worker threads получают `ContractError` до изменения баланса.
Методы синхронные, без await; проверка остатка, резерв и слот выполняются неделимо.

```python
from nsu_budget_agent.budget import BudgetLedger, budget_from_env

# Внутри async-функции; deadline и attempt_budget задаёт доверенный вызывающий код.
ledger = BudgetLedger(
    limit_units=budget_from_env(unit="competition_tokens"),  # Явный выбор, не автоопределение.
    deadline_monotonic=deadline,
)
ledger.open_attempt(attempt_id, attempt_budget)
lease = ledger.reserve(request_id, attempt_id, model, input_bound_tokens, output_cap_tokens)
ledger.mark_dispatched(request_id)  # До передачи запроса транспорту.
# После ответа: ledger.record_usage(normalized_usage).
```

Если запрос не отправлен, вызывающий код делает `release_unsent`. После отправки
при timeout/отмене он передаёт unknown usage; удержание остаётся полным. Даже
непосредственно `dispatched` уже удерживает резерв. После остановки/cleanup
вызывающий код делает `close_attempt`: неотправленные записи становятся released,
отправленные незавершённые — uncertain. Ledger сам не выполняет cleanup транспорта.

## Жизненный цикл и ошибки

- Переходы: reserved → dispatched/released; dispatched → uncertain/settled;
  uncertain → settled. Обратные переходы — `ContractError`.
- Повтор `open_attempt` с тем же бюджетом не обнуляет состояние и не открывает
  закрытую попытку. Повтор reserve с теми же параметрами возвращает текущий lease,
  даже после halt/закрытия; это чтение, не новая отправка. ID никогда не переиспользуется.
- Повторы dispatch/release в соответствующем состоянии идемпотентны. После dispatch
  release запрещён. Нулевой exact usage списывает ноль, но сохраняет использованный слот.
- Exact usage заменяет удержание расходом. Идентичный exact — no-op, unknown после
  exact игнорируется. Пока исход uncertain, сохраняется первый unknown для диагностики;
  более поздние частичные unknown-счётчики не меняют резерв и не считаются сверкой.
- Usage с чужим attempt/model, для неизвестного ID, released или до dispatch —
  `ContractError`. Неизвестные попытки и модели вне allowlist также отвергаются.
- Два разных exact вызывают `OperationFailed(accounting_violation)` и halt;
  первая exact-запись сохраняется до внешнего расследования. При exact выше резерва
  сначала записывается **полный** расход, затем такой же отказ/halt. Баланс не подрезается.
- Нехватка глобальных/локальных Units или слотов — `OperationFailed(budget_exhausted)`
  без новой записи. Истёкший дедлайн — `OperationFailed(timeout)`. Дедлайн проверяется
  при открытии, новом резерве и первой отправке. Дедлайн попытки выше общего — ContractError.
- `halt` сохраняет первую причину, запрещает только новые reserve/dispatch.
  Поздний usage, snapshot, release_unsent и close_attempt остаются доступны.

## Snapshot и границы гарантии

Каждый snapshot вычисляется из неизменяемых lease: spent — сумма exact,
held — полные reserved/dispatched/uncertain, calls — все записи кроме released.
Локальный snapshot использует лимит попытки, глобальный — общий лимит.
`available_units = max(0, limit - spent - held)`; расходы сверх лимита видны целиком.
Старые DTO-снимки не изменяются.

`blocked` означает, что нельзя начать новый LLM-запрос: halt, истёкший дедлайн,
менее 10 свободных Units (минимальный законный резерв), а для попытки также закрытие,
исчерпание слотов или отсутствие глобального остатка. Это не запрет локального решения
или сохранения ответа. Локальный `available_units` не заменяет глобальный snapshot.
Такая детализация blocked — явная интерпретация поля v1, а не новое поле контракта.

Операции reserve/snapshot линейны по числу записей. Нет дублированных накопительных
счётчиков, thread-lock, персистентности и восстановления после перезапуска.
«Одна попытка на задачу» обеспечивает будущий оркестратор: порт ledger не получает task_id.
Реальная гарантия расхода зависит от подтверждённых входных/выходных границ и usage
провайдера. Их нормализация и отключение retries относятся к A3, сохранение результатов — A5.

Проверки и ограничения: [отчёт A2](../../../docs/a2-report.md).
