# A5-lite runtime

`AgentRunner` владеет состояниями задач и выбирает `PlanDecision` существующим
Planner. `JiuwenAdapter` реализует неизменный AttemptEngine: последовательность
шагов исполняет реальный openJiuwen 0.1.19. После кандидата драйвер вызывает
Validator на отдельной копии, V1AcceptancePolicy и AnswerOnlyPublisher.

Доверенная точка сборки возвращает `runtime.Components(registry=..., tools=...,
validator=..., gateway=...)`. Gateway получает тот же Ledger и strict-конфигурацию.
`deterministic_solvers` задаются оператором и получают ноль LLM-вызовов.
Planner можно заменить через `Components.planner` без изменения Contracts v1.
Solver.step/ToolExecutor/Validator не должны создавать независимые циклы агента
или оставлять фоновые задачи после возврата/отмены. Их ОС-изоляция не обеспечивается SDK.

Полные команды запуска, тесты, единицы BUDGET и ограничения:
[docs/a5-report.md](../../../docs/a5-report.md).
