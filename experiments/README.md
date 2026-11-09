# Эксперименты и траектории

- `archive/` (в git не попадает): старые промежуточные прогоны, перенесены туда вместо удаления.
- `runs/` (в git не попадает): полные прогоны `tools/dev_run.sh`: копия задач с
  ответами, `logs/trajectories.jsonl`, `logs/summary.json`, `logs/stderr.log`.
- `results/`: то, что сохраняем: сводки и таблицы сравнения, небольшой пример
  траекторий.

Наборы задач:

| Набор | Откуда | Состав |
| --- | --- | --- |
| official8 | примеры организаторов (`case/task-dir`) | 2 math, 2 search, 2 data, 2 coding; все выполнимы |
| fresh50 | набор напарника (`../datasets/astra_fresh_50_*`), свой грейдер | 18 math, 16 search, 16 data; другие формулировки, многошаговые цепочки в базе знаний |
| practice80 | `tools/gen_tasks.py --n 80 --seed 11` | 4 типа по 20, ~20% невыполнимых, официальные шаблоны + перефразировки, coding с 1-3 багами |
| holdout100 | `tools/gen_tasks.py --n 100 --seed 23` | как practice80, на нём ничего не настраивалось |
| hard100 | `tools/gen_tasks.py --n 100 --seed 31 --hard` | 3-шаговые цепочки, подсчёты, несколько условий в CSV, 3-4 тонких бага, 2 типа невыполнимых, награды 1-100 |

| A, B, C | наборы напарника (`../datasets/astra_set_*_100_*`), свой грейдер, `tools/grade_astra.sh` | по 25 задач каждого типа, растущая сложность, ловушки, скрытые тесты coding |

Формат строки `trajectories.jsonl`:

```json
{"record": {"id": "...", "status": "answered|infeasible|failed|skipped", "answer": "...",
            "source": "fastpath|pass1:<model>|pass2:<model>", "estimate": 120.5, "cost": 113.2,
            "calls": 1, "seconds": 1.9, "attempts": ["pass1:<model>:answer"]},
 "events": [{"kind": "llm", "phase": "solve", "model": "...", "prompt_tokens": 431,
             "completion_tokens": 69, "cost": 112.1, "reply": "```python ...```"},
            {"kind": "exec", "what": "script", "returncode": 0, "stdout": "158"}]}
```
