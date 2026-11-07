"""Contracts v1 ports. Only SolverContext stores a service capability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import (
    AcceptanceDecision,
    AttemptBudget,
    AttemptResult,
    BudgetLease,
    BudgetSnapshot,
    ErrorInfo,
    Id,
    LLMReply,
    LLMRequest,
    LoadReport,
    ModelId,
    PlanDecision,
    PublicationPlan,
    RunConfig,
    SolverOffer,
    SolverStep,
    StepRecord,
    SubmissionReceipt,
    TaskProgress,
    TaskSpec,
    ToolCall,
    ToolContext,
    ToolResult,
    ToolSpec,
    UsageRecord,
    ValidationResult,
    _id,
    _instance,
    _int,
    _items,
    _require,
    _unique,
)


class LLMSession(Protocol):
    async def complete(self, request: LLMRequest) -> LLMReply: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class SolverContext:
    task: TaskSpec
    attempt_id: Id
    step_index: int
    budget: AttemptBudget
    tools: tuple[ToolSpec, ...]
    llm: LLMSession

    def __post_init__(self):
        _instance(self.task, TaskSpec, "task")
        _id(self.attempt_id, "attempt_id")
        _int(self.step_index, "step_index")
        _instance(self.budget, AttemptBudget, "budget")
        _require(self.step_index < self.budget.max_steps, "step_index exceeds budget")
        _items(self.tools, ToolSpec, "tools")
        _unique(tuple(tool.name for tool in self.tools), "tool name")
        # Protocols are structurally typed; no concrete gateway/SDK requirement.
        _require(callable(getattr(self.llm, "complete", None)), "llm must provide complete")


class TaskLoader(Protocol):
    async def load(self, tasks_dir: Path) -> LoadReport: ...


class TaskAdapter(Protocol):
    async def load_task(self, task_root: Path) -> TaskSpec: ...


class Solver(Protocol):
    @property
    def solver_id(self) -> str: ...

    def assess(self, task: TaskSpec, tools: tuple[ToolSpec, ...]) -> SolverOffer: ...

    async def step(self, ctx: SolverContext, history: tuple[StepRecord, ...]) -> SolverStep: ...


class SolverRegistry(Protocol):
    def register(self, solver: Solver) -> None: ...

    def all_solvers(self) -> tuple[Solver, ...]: ...

    def get(self, solver_id: str) -> Solver: ...


class Planner(Protocol):
    def choose(
        self,
        tasks: tuple[TaskSpec, ...],
        offers: tuple[SolverOffer, ...],
        progress: tuple[TaskProgress, ...],
        budget: BudgetSnapshot,
        config: RunConfig,
        now: float,
    ) -> PlanDecision | None: ...


class BudgetController(Protocol):
    def open_attempt(self, attempt_id: Id, budget: AttemptBudget) -> None: ...

    def reserve(
        self,
        request_id: Id,
        attempt_id: Id,
        model: ModelId,
        input_bound_tokens: int,
        output_cap_tokens: int,
    ) -> BudgetLease: ...

    def mark_dispatched(self, request_id: Id) -> BudgetLease: ...

    def release_unsent(self, request_id: Id) -> BudgetLease: ...

    def record_usage(self, usage: UsageRecord) -> BudgetLease: ...

    def snapshot(self, attempt_id: Id | None = None) -> BudgetSnapshot: ...

    def close_attempt(self, attempt_id: Id) -> None: ...

    def halt(self, error: ErrorInfo) -> None: ...


class LLMGateway(Protocol):
    def bind(self, attempt_id: Id, step_index: int, budget: AttemptBudget) -> LLMSession: ...


class ToolExecutor(Protocol):
    def catalog(self) -> tuple[ToolSpec, ...]: ...

    async def execute(self, call_id: Id, call: ToolCall, ctx: ToolContext) -> ToolResult: ...


class AttemptEngine(Protocol):
    async def run_attempt(
        self, task: TaskSpec, solver: Solver, budget: AttemptBudget
    ) -> AttemptResult: ...


class Validator(Protocol):
    async def validate(
        self, task: TaskSpec, result: AttemptResult, ctx: ToolContext
    ) -> ValidationResult: ...


class AcceptancePolicy(Protocol):
    def decide(self, result: AttemptResult, validation: ValidationResult) -> AcceptanceDecision: ...


class SubmissionLayout(Protocol):
    def plan(self, task: TaskSpec) -> PublicationPlan: ...


class Publisher(Protocol):
    async def publish(
        self, task: TaskSpec, result: AttemptResult, decision: AcceptanceDecision
    ) -> SubmissionReceipt: ...
