"""Stateless v1 solver capabilities; role sequencing belongs to JiuwenAdapter."""

from dataclasses import replace

from nsu_budget_agent.contracts.models import (
    ModelId,
    SolverOffer,
    SolverStep,
    StepRecord,
    StopAttempt,
    TaskSpec,
    ToolSpec,
)
from nsu_budget_agent.contracts.ports import SolverContext
from nsu_budget_agent.orchestration.decision import (
    TRANSITIONS,
    DecisionStore,
    DecisionTrace,
    Proposal,
    Review,
    Route,
    State,
    candidate_from,
    free_interpretation,
)

from .llm import MinimalLLMSolver
from .plans import explicit_missing_data
from .sources import selected_sources

__all__ = [
    "DecisionSolver",
    "FreeDecisionSolver",
    "DecisionStore",
    "DecisionTrace",
    "Proposal",
    "Review",
    "Route",
    "State",
    "TRANSITIONS",
    "free_interpretation",
]


class FreeDecisionSolver:
    solver_id = "local_router"

    def assess(self, task: TaskSpec, tools: tuple[ToolSpec, ...]) -> SolverOffer:
        call, complete, _, _ = free_interpretation(task)
        return SolverOffer(
            task_id=task.task_id,
            solver_id=self.solver_id,
            applicability="yes" if call and complete else "unknown",
            required_tools=(call.tool_name,) if call else (),
            expected_cost_units=0,
            reason="closed local grammar or free assessment",
        )

    async def step(self, ctx: SolverContext, history: tuple[StepRecord, ...]) -> SolverStep:
        try:
            selected_sources(ctx.task)
            call, complete, _, _ = free_interpretation(ctx.task)
            if call is None or not complete:
                return StopAttempt(status="unsupported", reason="full local grammar unavailable")
            if not history:
                return call
            return candidate_from(history[-1], call)
        except ValueError:
            return StopAttempt(status="blocked_by_evidence", reason="local evidence unavailable")


class DecisionSolver:
    """A normal v1 consumer gets only independently verifiable one-role actions.

    A5's internal controller expands the strategy using its own per-run state;
    this capability never stores that state or changes StopAttempt semantics.
    """

    solver_id = "generic"

    def __init__(self, model: ModelId) -> None:
        self.model = model

    def assess(self, task: TaskSpec, tools: tuple[ToolSpec, ...]) -> SolverOffer:
        return SolverOffer(
            task_id=task.task_id,
            solver_id=self.solver_id,
            applicability="no"
            if task.type_hint == "coding"
            or task.check_hint == "unittest"
            or explicit_missing_data(task)
            else "unknown",
            required_tools=(),
            reason="A5 bounded roles; unknown estimates",
        )

    async def step(self, ctx: SolverContext, history: tuple[StepRecord, ...]) -> SolverStep:
        missing = explicit_missing_data(ctx.task)
        if missing:
            return StopAttempt(status="blocked_by_evidence", reason=missing)
        try:
            sources = selected_sources(ctx.task)
        except ValueError:
            return StopAttempt(status="blocked_by_evidence", reason="sensitive source is forbidden")
        safe_context = replace(ctx, task=replace(ctx.task, artifacts=sources))
        return await MinimalLLMSolver(self.model).step(safe_context, history)
