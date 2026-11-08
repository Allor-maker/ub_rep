"""Offline production default and explicit strict LLM composition for A5."""

import json
import os
from collections.abc import Mapping
from dataclasses import replace
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
    ProviderProfile,
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


def build_production(
    ledger: BudgetController,
    config: RunConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    environ: Mapping[str, str] | None = None,
) -> Components:
    """No key means offline; RunConfig selects strict or controlled-risk competition."""
    environment = os.environ if environ is None else environ
    if not environment.get("LLM_API_KEY", "").strip():
        return build(ledger, config)
    if config.accounting_mode == "best_effort":
        return build_competition(ledger, config, transport=transport, environ=environment)
    if (
        not environment.get("LLM_BASE_URL", "").strip()
        or not environment.get("LLM_COMPATIBILITY_ADAPTER", "").strip()
    ):
        raise ContractError("LLM activation requires URL and a reviewed compatibility adapter")
    return build_with_router(ledger, config, transport=transport, environ=environment)


def build_competition(
    ledger: BudgetController,
    config: RunConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    environ: Mapping[str, str] | None = None,
) -> Components:
    """A7.1 through the existing A3/A2, using estimates without attestations."""
    from nsu_budget_agent.llm import NeuralDeepGateway
    from nsu_budget_agent.llm.competition import estimate_input, wire_body
    from nsu_budget_agent.orchestration.decision import DecisionController
    from nsu_budget_agent.planning.decision import DecisionPortfolioPlanner
    from nsu_budget_agent.validation.decision import DecisionValidator

    from .decision import DecisionSolver, DecisionStore, FreeDecisionSolver

    if config.accounting_mode != "best_effort":
        raise ContractError("competition requires explicit best_effort accounting")
    environment = os.environ if environ is None else environ
    model_name = environment.get("LLM_MODEL", "qwen3.6-35b-a3b-noreason")
    review_name = environment.get("LLM_REVIEW_MODEL", model_name)
    if model_name not in get_args(ModelId) or review_name not in get_args(ModelId):
        raise ContractError("invalid competition model")
    model, review_model = cast(ModelId, model_name), cast(ModelId, review_name)
    try:
        output_cap = int(environment.get("LLM_OUTPUT_CAP", "512"))
    except ValueError as exc:
        raise ContractError("invalid competition output cap") from exc
    if not 64 <= output_cap <= 2048:
        raise ContractError("competition output cap must be in [64, 2048]")
    try:
        http_timeout = float(environment.get("LLM_HTTP_TIMEOUT_S", "45"))
    except ValueError as exc:
        raise ContractError("invalid competition HTTP timeout") from exc
    if not 1 <= http_timeout <= 180:
        raise ContractError("competition HTTP timeout must be in [1, 180]")

    def reservation_estimate(request: LLMRequest) -> int:
        return estimate_input(request, wire_body(request)) + 10 * request.max_output_tokens

    store, tools = DecisionStore(), LocalToolExecutor()
    controller = DecisionController(
        store,
        model,
        review_model=review_model,
        reserve_mandatory_calls=True,
        output_cap=output_cap,
        reservation_bound=reservation_estimate,
        require_independent=environment.get("LLM_REQUIRE_INDEPENDENT", "0") == "1",
        length_retry=environment.get("LLM_LENGTH_RETRY", "0") == "1",
    )
    registry = StaticSolverRegistry()
    registry.register(FreeDecisionSolver())
    registry.register(DecisionSolver(model))
    gateway = NeuralDeepGateway(
        budget=ledger,
        input_bound=estimate_input,
        profiles={m: ProviderProfile() for m in get_args(ModelId)},
        accounting_mode="best_effort",
        competition_mode=True,
        timeout_s=http_timeout,
        transport=transport,
        environ=environment,
    )
    return Components(
        registry=registry,
        tools=tools,
        gateway=gateway,
        validator=DecisionValidator(tools, store),
        deterministic_solvers=("local_router",),
        decision_store=store,
        decision_controller=controller,
        planner=DecisionPortfolioPlanner(
            tools=tools.catalog(), deterministic_solvers=("local_router",), controller=controller
        ),
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


def build_with_router(
    ledger: BudgetController,
    config: RunConfig,
    *,
    compatibility: "ProviderCompatibility | None" = None,
    transport: httpx.AsyncBaseTransport | None = None,
    environ: Mapping[str, str] | None = None,
) -> Components:
    """A7.1 opt-in; reuse A3.5 strict setup without changing legacy factories."""
    from nsu_budget_agent.llm.compatibility import load_compatibility
    from nsu_budget_agent.orchestration.decision import DecisionController
    from nsu_budget_agent.planning.decision import DecisionPortfolioPlanner
    from nsu_budget_agent.validation.decision import DecisionValidator

    from .decision import DecisionSolver, DecisionStore, FreeDecisionSolver
    from .llm import MinimalLLMSolver

    environment = os.environ if environ is None else environ
    adapter = (
        compatibility
        if compatibility is not None
        else load_compatibility(environment.get("LLM_COMPATIBILITY_ADAPTER", ""))
    )
    base = build_with_llm(
        ledger, config, compatibility=adapter, transport=transport, environ=environ
    )
    previous = base.registry.get("generic")
    assert isinstance(previous, MinimalLLMSolver)
    store = DecisionStore()

    def reservation_bound(request: LLMRequest) -> int:
        body = json.dumps(
            {
                "model": request.model,
                "messages": [{"role": m.role, "content": m.content} for m in request.messages],
                adapter.output_token_parameter: request.max_output_tokens,
                "stream": False,
                "n": 1,
            },
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        bound = adapter.input_bound(request, body)
        if type(bound) is not int or bound < 0:
            raise ContractError("invalid trusted input bound")
        return bound + 10 * request.max_output_tokens

    controller = DecisionController(store, previous.model, reservation_bound=reservation_bound)
    registry = StaticSolverRegistry()
    registry.register(FreeDecisionSolver())
    registry.register(DecisionSolver(previous.model))
    return replace(
        base,
        registry=registry,
        deterministic_solvers=("local_router",),
        decision_store=store,
        decision_controller=controller,
        validator=DecisionValidator(base.tools, store),
        planner=DecisionPortfolioPlanner(
            tools=base.tools.catalog(),
            deterministic_solvers=("local_router",),
            controller=controller,
        ),
    )
