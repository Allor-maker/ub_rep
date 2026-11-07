# A1: общие определения Contracts v1

Проверено 7 ноября 2026. База работы: `c6ba29e` (`main`, синхронизирована
через `git pull --ff-only`), ветка `feature/a1-contracts`.

## Что требовалось и что реализовано

A1 даёт обоим разработчикам единые типы данных и интерфейсы по уже принятой
спецификации [Contracts v1](contracts.md). B может писать инструменты, решатель
и validator независимо от бюджета, сети и SDK A.

- `contracts/models.py`: 33 DTO, алиасы типов, `ContractError`, `OperationFailed`
  и проверка инвариантов. `SolverContext` — 34-й DTO, находится в `ports.py`.
- `contracts/ports.py`: все 14 Protocol v1 с установленными sync/async-границами.
  `models.py` не импортирует `ports.py`. Общий слой использует только stdlib.
- Все DTO — frozen/slots/kw_only dataclasses, коллекции — tuple. Открытые метки
  типов и неизвестные коды ошибок сохраняются. `None` не становится нулём.
- Проверяются числовые границы, bool вместо чисел, конечность времени, модели,
  ID, хеши, уникальность путей и согласованность связанных состояний/идентификаторов.
  Полный расход вычисляется как `input + 10 * output`, reasoning повторно не прибавляется.
- Вложенный JSON копируется и замораживается рекурсивно. Внутренние подклассы
  dict/list сохраняют JSON-сериализацию и совместимость с `dataclasses.asdict`;
  обычные операции изменения запрещены. Callable, циклы, нестроковые ключи,
  NaN/Infinity и внешние `$ref` в схемах отвергаются.
- `parse_rel_path(raw)` проверяет **исходную строку до нормализации** и возвращает
  PurePosixPath. Все внешние строки путей должны проходить через этот помощник.
  Уже созданный `PurePosixPath('a/./b')` теряет сведения о `./`; восстановить их
  в конструкторе DTO невозможно. `allow_directory=True` разрешает только отдельный
  `.` для аргумента каталога инструмента; файловые DTO его не принимают.
- `tests/contracts/`: C01, LLMSession-double и пример структурной совместимости
  реализаций B (`typecheck_consumers.py`). Сервисы A2–A6 здесь не реализуются.

Проверки, которым нужны состояние процесса и ФС, остаются у соответствующих
сервисов: symlink/каноничность/доступность корней, защищённые и служебные файлы,
лимит размера кандидата, существование evidence, пределы назначения, дедлайн
при старте, переходы ledger и монотонность защиты. DTO не выполняют файловый I/O
и не подменяют проверки C02–C12. Политика принятия и публикация — A5.

## Как проверить общий слой без установки пакетов

Из корня репозитория, Python 3.11, PowerShell:

```powershell
$env:PYTHONPATH = 'src'
python -S -m unittest discover -s tests -v
```

Результат: **22 теста успешно**, включая подслучаи для всех 34 DTO.
Проверяются frozen/slots/kw_only, обязательная явная неизвестность, bool/int,
некорректные значения состояний и чисел, пути до нормализации, глубоко вложенный
JSON, связи usage/lease/reply, candidate/receipt, агрегация validation и async-порты.
Отдельный subprocess импортирует contracts с `-S`, без сторонних пакетов;
audit-hook отвергает сетевые обращения. Тесты не используют данные организаторов.

Необязательные инструменты разработки:

```powershell
python -m venv .venv-a1
.\.venv-a1\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv-a1\Scripts\python.exe -m ruff check src tests
.\.venv-a1\Scripts\python.exe -m ruff format --check src tests
$env:MYPYPATH = 'src'
.\.venv-a1\Scripts\python.exe -m mypy src/nsu_budget_agent/contracts tests/contracts/typecheck_consumers.py
```

Результат: ruff check/format и mypy успешны. Статический пример B не наследует
конкретные сервисы: обычные классы удовлетворяют Solver/ToolExecutor/Validator.

## Отдельная проверка openJiuwen

[Релиз openJiuwen 0.1.19](https://pypi.org/project/openjiuwen/0.1.19/) выбран
по спецификации. Соседний `my_env` содержит 0.1.20.post1 и для этой проверки
не использовался. Создана отдельная игнорируемая `.venv-a1`.

```powershell
.\.venv-a1\Scripts\python.exe -m pip install -r requirements-sdk-probe.txt
.\.venv-a1\Scripts\python.exe -m pip check
$env:PYTHONPATH = 'src'
.\.venv-a1\Scripts\python.exe -m nsu_budget_agent.orchestration.sdk_probe
```

Фактическая среда: **Windows x86_64, CPython 3.11.0, openJiuwen 0.1.19**.
Установка из PyPI и `pip check` успешны. Smoke завершился с кодом 0:

```json
{"python":"3.11.0","platform":"Windows","openjiuwen":"0.1.19","two_components":"passed","cancel_cleanup":"passed","deadline_cleanup":"passed","external_network":"blocked"}
```

Реальный SDK импортирован только в `orchestration/sdk_probe.py`. Проверены
`Workflow`, `WorkflowCard`, `WorkflowComponent.invoke`, `Start`, `End`,
`create_workflow_session`, последовательное исполнение двух компонентов через
`Workflow.invoke`, проброс CancelledError, `asyncio.timeout_at`, выполнение finally
компонента и `Runner.stop`. У компонентов явно `max_retries=0`.
Проверка блокирует внешние соединения/DNS, разрешая loopback для asyncio Windows;
модели не загружаются, API/ключи не используются. Логи SDK могут содержать ERROR
при ожидаемой отмене: успех определяют assertions и итоговый код процесса.

Это эксперимент совместимости API, а не JiuwenAdapter/AttemptEngine и не C12.
Ограничение одного complete на шаг, отключение повторов транспорта и работа
реальных внешних портов будут проверены в A3/A5/C07/C12.

## Воспроизводимость и оставшиеся ограничения

`requirements-sdk-probe-windows-py311.lock` фиксирует точные версии всех 168 установленных
пакетов, включая pip/setuptools и dev-инструменты. Это снимок Windows-эксперимента без
хешей, не финальный runtime-lock. Для повторения на Windows/Python 3.11:

```powershell
python -m venv .venv-a1-repro
.\.venv-a1-repro\Scripts\python.exe -m pip install --no-compile -r requirements-sdk-probe-windows-py311.lock
.\.venv-a1-repro\Scripts\python.exe -m pip check
$env:PYTHONPATH = 'src'
.\.venv-a1-repro\Scripts\python.exe -m nsu_budget_agent.orchestration.sdk_probe
```

Повторная установка этого снимка в чистую `.venv-a1-repro` **проверена успешно**:
все 168 имён/версий совпадают с файлом, `pip check` успешен, SDK probe повторно
прошёл все три сценария с кодом 0 и блокировкой внешней сети.
Установка только `requirements-sdk-probe.txt` фиксирует SDK, но допускает новые
версии транзитивных зависимостей. Обычные C01 не требуют ни одного из этих файлов.
`pip install .` пока не поддерживается: pyproject содержит настройки инструментов,
упаковка runtime относится к A6.

Linux-проверка A1 **остаётся открытой**: Docker в среде отсутствует; зарегистрированная
WSL Ubuntu-24.04 не запускается из-за отсутствующего `ext4.vhdx`
(`Wsl/Service/CreateInstance/MountDisk/HCS/ERROR_PATH_NOT_FOUND`). Она не отмечена
успешной по Windows-результату. В рабочем Linux/Python 3.11 нужно повторить probe,
зафиксировать платформенные зависимости и проверить чистую установку; финальные
контейнер, образ, размер и ограничения проверяет A6. Неисправная WSL не изменялась.

Готов к review B общий stdlib-контракт A1 и C01. Ветка локальная, commit/push/PR
не создавались. A1 целиком не следует закрывать до проверки Linux; A2 и работы B
по общим определениям уже могут начинаться.
