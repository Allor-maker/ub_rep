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

## A4

`unit/test_task_loader.py`, `unit/test_planner.py` и `contracts/test_a4_planning.py`
проверяют read-only загрузку, безопасный ввод, реестр, чистое планирование и coverage
с fake-компонентами B. `contracts/typecheck_planning.py` проверяет неизменные порты.
Полный discovery CI включает новые тесты автоматически; POSIX и symlink проверки
могут быть пропущены на Windows и должны выполниться в Linux CI.
