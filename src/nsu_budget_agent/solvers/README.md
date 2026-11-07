# Решатели — B

`DeterministicSolver(kind)` реализует stateless assess/step для math/search/data_analysis.
`plans.py` разбирает только явную узкую грамматику, независимо от ID задач и названий
тренировочных файлов. assess — без I/O; step возвращает ToolCall, затем CandidateAnswer
с evidence либо StopAttempt. Внешний цикл исполняется A5/openJiuwen.

`composition.build(ledger, config)` собирает production Components A5 с LocalToolExecutor
и LocalValidator. Generic честно возвращает unsupported; LLM backend недоступен,
не симулируется и не создаёт запросов. Все маршруты помечены deterministic, cost=0;
probability и duration не выдумываются. Coding и model fallback не реализованы.

Поддержанные условия и результаты: [B-minimal](../../../docs/b-minimal-report.md).
