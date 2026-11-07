"""Static compatibility example for B; mypy checks structural Protocol matching."""

from nsu_budget_agent.contracts.models import (
    AttemptResult,
    CandidateAnswer,
    CheckResult,
    Id,
    SolverOffer,
    SolverStep,
    StepRecord,
    TaskSpec,
    ToolCall,
    ToolContext,
    ToolResult,
    ToolSpec,
    ValidationResult,
)
from nsu_budget_agent.contracts.ports import Solver, SolverContext, ToolExecutor, Validator


class ExampleSolver:
    @property
    def solver_id(self) -> str:
        return "generic"

    def assess(self, task: TaskSpec, tools: tuple[ToolSpec, ...]) -> SolverOffer:
        return SolverOffer(
            task_id=task.task_id,
            solver_id=self.solver_id,
            applicability="unknown",
            required_tools=(),
        )

    async def step(self, ctx: SolverContext, history: tuple[StepRecord, ...]) -> SolverStep:
        return CandidateAnswer(text="example")


class ExampleTools:
    def catalog(self) -> tuple[ToolSpec, ...]:
        return ()

    async def execute(self, call_id: Id, call: ToolCall, ctx: ToolContext) -> ToolResult:
        return ToolResult(call_id=call_id, tool_name=call.tool_name, status="ok")


class ExampleValidator:
    async def validate(
        self, task: TaskSpec, result: AttemptResult, ctx: ToolContext
    ) -> ValidationResult:
        assert result.snapshot_sha256 is not None
        return ValidationResult(
            attempt_id=result.attempt_id,
            snapshot_sha256=result.snapshot_sha256,
            checks=(
                CheckResult(
                    check_id="example", status="inconclusive", blocking=True, message="Test double"
                ),
            ),
            status="inconclusive",
        )


solver: Solver = ExampleSolver()
tools: ToolExecutor = ExampleTools()
validator: Validator = ExampleValidator()
