"""Trusted offline production assembly for the existing A5 Components factory."""

from nsu_budget_agent.contracts.models import (
    AttemptBudget,
    ErrorInfo,
    LLMReply,
    LLMRequest,
    OperationFailed,
    RunConfig,
)
from nsu_budget_agent.contracts.ports import BudgetController, LLMSession
from nsu_budget_agent.orchestration.runtime import Components
from nsu_budget_agent.planning import StaticSolverRegistry
from nsu_budget_agent.tools import LocalToolExecutor
from nsu_budget_agent.validation import LocalValidator

from .deterministic import DeterministicSolver, UnsupportedGenericSolver


class DisabledLLMGateway:
    """Production capability denial, not a simulated provider or canned response."""

    def bind(self, attempt_id: str, step_index: int, budget: AttemptBudget) -> LLMSession:
        return self

    async def complete(self, request: LLMRequest) -> LLMReply:
        raise OperationFailed(ErrorInfo(code="unsupported", message="B-minimal has no LLM backend"))


def build(ledger: BudgetController, config: RunConfig) -> Components:
    registry = StaticSolverRegistry()
    registry.register(UnsupportedGenericSolver())
    for kind in ("math", "search", "data_analysis"):
        registry.register(DeterministicSolver(kind))
    tools = LocalToolExecutor()
    return Components(
        registry=registry,
        tools=tools,
        validator=LocalValidator(tools),
        gateway=DisabledLLMGateway(),
        deterministic_solvers=tuple(s.solver_id for s in registry.all_solvers()),
    )
