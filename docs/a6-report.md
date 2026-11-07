# A6 — первый конкурсный Docker baseline

Реализовано в `feature/a6-docker`: Dockerfile, `.dockerignore`, прямые runtime
requirements и smoke harness для production B-minimal + настоящего openJiuwen
0.1.19. Docker-образ и Linux ещё не проверены запуском в этом окружении; это
ограничение результатов ниже, а не подтверждённая готовность контейнера.

## Состав и конкурсный CLI

- Многоэтапный `Dockerfile` использует `python:3.11.11-slim-bookworm`, создаёт
  `/opt/venv` и во время сборки устанавливает `openjiuwen==0.1.19` и
  `httpx==0.28.1` с их зависимостями. `pip check` выполняется при сборке.
  Отдельная стадия собирает `/app` только из runtime Python-файлов; runtime-стадия
  не выполняет pip install.
- `.dockerignore` ограничивает build context runtime-файлами. Тесты репозитория,
  benchmark, expected/gold, локальные окружения/секреты, документация и SDK probe
  в `/app` не копируются. Из venv удаляются каталоги и Python-модули тестов и
  байткод сторонних пакетов; после этого работоспособность SDK должна подтвердиться
  Linux smoke. Production-образ не использует test/fake factory.
- Команда образа и вызов организатора: `python /app/run_agent.py --tasks-dir /data/tasks`.
  CLI выбирает существующую production factory B-minimal. Нет LLM fallback, coding
  publication или A7; реальные решатели ограничены [отчётом B](b-minimal-report.md).
- `--budget-unit` теперь по умолчанию `competition_tokens` по явному решению для
  этого baseline. `BUDGET=100000` преобразуется в **1 000 000 внутренних Units**;
  стоимость `prompt_tokens + 10 * completion_tokens`, вход/10 + выход во внешних
  единицах. Это не меняет Ledger/Contracts v1. Явный `--budget-unit units_x10`
  сохраняется и при том же BUDGET задаёт 100 000 Units. Преобразование дробного
  competition budget остаётся floor(10 × BUDGET). Если BUDGET отсутствует или
  некорректен, запуск отказывает. Вопрос Q1 об организаторских единицах ещё
  требует подтверждения; выбранный default — заданное пользователем предположение.
- JSON CLI теперь содержит `limit_units` из фактического снимка A2, а также
  `spent_units` и `held_units`. Нулевой бюджет разрешает детерминированные
  маршруты без вызовов модели. Strict-mode A3 не изменён и не выключался.

## Ограниченная файловая система и сеть

Рабочий каталог образа `/tmp`: в SDK 0.1.19 часть относительных логов использует
текущий каталог. `HOME`, `TMPDIR`, `XDG_CACHE_HOME`, `HF_HOME` и `MPLCONFIGDIR`
направлены в `/tmp`; `PYTHONDONTWRITEBYTECODE=1`, offline-переменные Hugging Face
включены. `/data/tasks` — каталог для внешнего writable mount. Временный файл
публикации создаётся в каталоге самой задачи и связывается там же, поэтому граница
между `/tmp` tmpfs и `/data` не приводит к cross-device hard link.

Production B не делает запросов к NeuralDeep и не требует ключей. Отсутствие
скачиваний проверяется планируемым Docker smoke с `--network none`; переменные
offline сами по себе не являются ОС-сетевой изоляцией. Принудительная изоляция
недоверенного кода и coding-инструмент в baseline отсутствуют. Источники gold с
произвольными именами должны быть физически отделены от input.

Текущие параметры запуска: общий deadline по умолчанию 3300 с, до 30 с на
попытку, одна задача одновременно. Конкурсная команда использует 3300 с,
оставляя запас относительно лимита 3600 с. Общий deadline можно явно задать
через `--timeout-s` (максимум 3600 с); лимит попытки по-прежнему равен
меньшему из 30 с и общего deadline.

## Сборка и проверка Docker на Linux

Из корня репозитория, при установленном и работающем Docker:

```bash
start=$(date +%s)
docker build --platform linux/amd64 --tag nsu-budget-agent:a6 .
elapsed=$(($(date +%s) - start))
bytes=$(docker image inspect --format '{{.Size}}' nsu-budget-agent:a6)
printf 'build seconds=%s; uncompressed image bytes=%s\n' "$elapsed" "$bytes"
test "$elapsed" -lt 1000
test "$bytes" -lt 4000000000
```

Измеряется полное время build, включая получение образа/зависимостей, и
несжатый размер image, а не размер context или сумма размеров колёс. Порог в
4 000 000 000 байт консервативно соответствует ограничению <4 GB. Сборка
разрешает зависимости в Linux, Windows lock не используется.

Переносимый Docker smoke создаёт шесть собственных задач и запускает фактически
конкурсную команду с BUDGET=0:

```bash
python tests/docker/smoke.py --mode docker --image nsu-budget-agent:a6
```

Для уже имеющегося публичного smoke benchmark из 12 задач:

```bash
python tests/docker/smoke.py --mode docker --image nsu-budget-agent:a6 \
  --source /absolute/path/to/nsu_benchmark_smoke_12_tasks/tasks
```

Harness использует A4 Loader и открывает только разрешённые публичные artifacts.
В собственный временный каталог экспортируются публичные поля и файлы: исходный
manifest целиком, expected/gold, старые ответы и незагруженные данные не копируются.
Исходный набор не изменяется. Ожидаемые ответы вычисляются отдельными тестовыми
формулами по публичным данным; для нерешённых задач проверяется отсутствие файла.
Сравнение BUDGET=0, 0 spent/held, числа созданных ответов и вызовов SDK проводится
автоматически. Для Docker Desktop временный каталог должен быть доступен демону;
при необходимости задайте `--work-parent /absolute/shared/path`.

Docker smoke дополнительно проверяет содержимое `/app` и отсутствие test-каталогов,
test Python-модулей и bytecode в venv. Запуск контейнера выполняется с
`--read-only`, `--network none`, `--cap-drop ALL`, `--pids-limit 512`, `--cpus 4`,
`--memory 4g`, tmpfs `/tmp` на 2 GiB и writable bind mount задач. На POSIX он
передаёт текущие UID:GID, чтобы созданная временная директория оставалась доступной.
Это smoke на ограниченном наборе, а не доказательство отсутствия всех исходящих
операций в других конфигурациях.

Ручной эквивалент при уже подготовленной **свежей** копии входных задач:

```bash
docker run --rm --read-only --network none --cap-drop ALL --pids-limit 512 \
  --cpus 4 --memory 4g --tmpfs /tmp:rw,size=2g,mode=1777 \
  --user "$(id -u):$(id -g)" --env BUDGET=0 \
  --mount "type=bind,src=/absolute/path/to/fresh/tasks,dst=/data/tasks" \
  nsu-budget-agent:a6 python /app/run_agent.py --tasks-dir /data/tasks
```

UID/GID запуска должен иметь право записи в каталог задач. Образ по умолчанию
использует `/tmp` как cwd; если организатор переопределит рабочий каталог, он
также должен быть writable для SDK-логов либо потребуется отдельная конфигурация
логирования. Базовый образ не запускает API или тестовые компоненты автоматически.

## Фактические результаты

В текущем окружении Windows Docker и Podman в PATH не обнаружены. `wsl --status`
завершился с `Wsl/EnumerateDistros/Service/E_ACCESSDENIED`. Поэтому **docker build,
docker run, фактический размер образа, build time и Linux/readonly-rootfs/network-none
не проверены локально**. Не утверждается, что образ собирается или проходит Linux
до успешного GitHub Actions/доступного Docker. В CI добавлен отдельный ubuntu-latest
job: build linux/amd64 с timeout 1000 с, проверка image size и Docker smoke выше.
Job ещё не запускался; секреты, NeuralDeep и Windows lock не используются.

Фактически выполнено на Windows/Python 3.11:

- Production CLI на свежей публичной копии локального smoke benchmark с BUDGET=0,
  без --budget-unit: **6/12** проверенных по публичным данным ответов, для остальных
  answer.txt отсутствует; 12 реальных SDK workflows, spent=held=0. Это host smoke,
  не Docker/Linux test. Оригинальный каталог не менялся. Переносимые собственные
  фикстуры: **3/6** на host, нерешённые без answer.txt.
- Конкурсный CLI без --budget-unit и BUDGET=100000: снимок реального Ledger
  `limit_units=1000000`, сохранён ответ, SDK workflow выполняется. Явный override
  units_x10 при том же BUDGET даёт 100000 внутренних Units.
- Полный набор после изменения deadline: **206 тестов, OK (skipped=6)**, то есть
  200 выполнено и 6 пропущено
  по Windows-платформе/праву на symlink. Ruff lint и format-check проходят;
  Mypy — без ошибок в 45 файлах по команде CI. Выполнены статические проверки
  YAML, Python-команды проверки payload, Dockerfile CMD, base tag, прямых pin и cwd;
  это не сборка Docker. `pip check` в существующем Windows venv прошёл.
- Размер tracked и неигнорируемых файлов рабочего дерева без `.git`: 799 537 байт
  на момент проверки, ниже 100 MB. Это размер файлов репозитория, не размер образа;
  локальные игнорируемые `.venv`/логи в него не входят и в build context запрещены.

Host smoke можно воспроизвести без Docker (нужен установленный SDK в Python 3.11):

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv-a1\Scripts\python.exe tests/docker/smoke.py --mode host --source C:\VSCode\openJiuwen\nsu_benchmark_smoke_12_tasks\tasks
```

Без `--source` используются шесть переносимых собственных задач. На Windows harness
по умолчанию создаёт временную копию внутри корня репозитория, чтобы безопасное
открытие предков каталога работало в ограниченной среде; после прогона копия
удаляется. Никакие эталоны пользовательского набора не читаются.

## Оставшиеся риски

- Зафиксированы Python patch/OS family в теге и версии двух прямых зависимостей;
  тег не закреплён digest, транзитивные версии не зафиксированы Linux lock с hashes.
  Поэтому команды повторяемы, но побитовая воспроизводимость/одинаковый размер
  разных сборок не гарантированы. Нужен проверенный Linux lock и digest после
  доступного Docker/build-среды; Windows-specific lock не подходит.
- Размер/время, разрешение зависимостей, совместимость wheels и системных библиотек,
  а также влияние удаления тестовых файлов сторонних пакетов в Linux не подтверждены.
  CI обязан подтвердить не только pip check, но и реальный SDK workflow после prune.
- Только B-minimal: узкая грамматика, нет coding или LLM fallback; результат smoke
  не является конкурсной оценкой. Возможны неоптимальные отказы на новых заданиях.
- Реальный прокси и strict-профили A3 не проверены, единицы BUDGET и формат сдачи
  требуют подтверждения организаторов. Нет обещания прохождения отбора или 100 задач.
- Остаточная гонка внешнего писателя исходников и необходимость поддержки hard links
  описаны в [A5](a5-report.md). Публикация не перезаписывает существующий answer.txt.
