"""Offline production default and explicit strict LLM composition for A5."""

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast, get_args

import httpx

from nsu_budget_agent.contracts.models import (
    AttemptBudget,
    ContractError,
    ErrorInfo,
    LLMReply,
    LLMRequest,
    ModelId,
    OperationFailed,
    RunConfig,
)
from nsu_budget_agent.contracts.ports import BudgetController, LLMSession
from nsu_budget_agent.orchestration.runtime import Components
from nsu_budget_agent.planning import StaticSolverRegistry
from nsu_budget_agent.tools import LocalToolExecutor
from nsu_budget_agent.validation import LocalValidator

from .deterministic import DeterministicSolver, UnsupportedGenericSolver

if TYPE_CHECKING:
    from nsu_budget_agent.llm.compatibility import ProviderCompatibility


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


def build_with_llm(
    ledger: BudgetController,
    config: RunConfig,
    *,
    compatibility: "ProviderCompatibility | None" = None,
    transport: httpx.AsyncBaseTransport | None = None,
    environ: Mapping[str, str] | None = None,
) -> Components:
    """Opt in via --components nsu_budget_agent.solvers.composition:build_with_llm.

    The adapter is trusted deployment code. No default real-provider attestation
    is shipped; neither keys nor a successful smoke enable this factory alone.
    """
    from nsu_budget_agent.llm import NeuralDeepGateway
    from nsu_budget_agent.llm.compatibility import ProviderCompatibility, load_compatibility
    from nsu_budget_agent.validation.llm import FallbackValidator

    from .llm import MinimalLLMSolver

    environment = os.environ if environ is None else environ
    if config.accounting_mode != "strict":
        raise ContractError("LLM fallback requires strict accounting")
    model_name = environment.get("LLM_MODEL", "qwen3.6-35b-a3b-noreason")
    if model_name not in get_args(ModelId):
        raise ContractError("invalid fallback model")
    model = cast(ModelId, model_name)
    adapter = (
        compatibility
        if compatibility is not None
        else load_compatibility(environment.get("LLM_COMPATIBILITY_ADAPTER", ""))
    )
    if not isinstance(adapter, ProviderCompatibility):
        raise ContractError("invalid compatibility adapter")
    adapter.require_verified((model,))
    registry = StaticSolverRegistry()
    registry.register(MinimalLLMSolver(model))
    deterministic = tuple(DeterministicSolver(kind) for kind in ("math", "search", "data_analysis"))
    for solver in deterministic:
        registry.register(solver)
    tools = LocalToolExecutor()
    gateway = NeuralDeepGateway(
        budget=ledger,
        input_bound=adapter.input_bound,
        profiles=adapter.profiles,
        output_token_parameter=adapter.output_token_parameter,
        accounting_mode="strict",
        transport=transport,
        environ=environment,
    )
    return Components(
        registry=registry,
        tools=tools,
        validator=FallbackValidator(tools),
        gateway=gateway,
        deterministic_solvers=tuple(s.solver_id for s in deterministic),
    )
