"""Sequential A5-lite driver. Components B are supplied by trusted composition."""

import asyncio
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .decision import DecisionController, DecisionStore, DecisionTrace

from nsu_budget_agent.contracts.models import (
    AttemptResult,
    BudgetSnapshot,
    CheckResult,
    ContractError,
    ErrorInfo,
    LoadReport,
    RunConfig,
    SubmissionReceipt,
    TaskProgress,
    TaskSpec,
    ToolContext,
    ValidationResult,
)
from nsu_budget_agent.contracts.ports import (
    BudgetController,
    LLMGateway,
    Planner,
    SolverRegistry,
    ToolExecutor,
    Validator,
)
from nsu_budget_agent.planning import BaselinePlanner, next_not_before_wakeup
from nsu_budget_agent.submission import AnswerOnlyPublisher, V1AcceptancePolicy
from nsu_budget_agent.task_io import DirectoryTaskLoader
from nsu_budget_agent.task_io.workspace import WorkspaceStore, joined_io

from .engine import JiuwenAdapter


@dataclass(frozen=True, kw_only=True)
class Components:
    registry: SolverRegistry
    tools: ToolExecutor
    validator: Validator
    gateway: LLMGateway
    deterministic_solvers: tuple[str, ...] = ()
    planner: Planner | None = None
    decision_store: "DecisionStore | None" = None
    decision_controller: "DecisionController | None" = None


ComponentFactory = Callable[[BudgetController, RunConfig], Components]


@dataclass(frozen=True, kw_only=True)
class AssessmentFailure:
    task_id: str
    solver_id: str
    error: ErrorInfo


@dataclass(frozen=True, kw_only=True)
class RunReport:
    loaded: LoadReport
    progress: tuple[TaskProgress, ...]
    attempts: tuple[AttemptResult, ...]
    receipts: tuple[SubmissionReceipt, ...]
    budget: BudgetSnapshot
    workflow_invocations: int
    component_steps: int
    assessment_errors: tuple[AssessmentFailure, ...] = ()
    decision_traces: tuple["DecisionTrace", ...] = ()
    validations: tuple[ValidationResult, ...] = ()


class AgentRunner:
    def __init__(
        self,
        *,
        config: RunConfig,
        ledger: BudgetController,
        components: Components,
        staging_parent: Path | None = None,
    ) -> None:
        if components.decision_controller is not None:
            from nsu_budget_agent.planning.decision import DecisionPortfolioPlanner

            from .decision import DecisionController, DecisionStore

            store = DecisionStore()
            previous = components.decision_controller
            controller = DecisionController(
                store,
                previous.model,
                output_cap=previous.output_cap,
                reservation_bound=previous.reservation_bound,
                review_model=previous.review_model,
                reserve_mandatory_calls=previous.reserve_mandatory_calls,
                require_independent=previous.require_independent,
                length_retry=previous.length_retry,
            )
            validator = getattr(components.validator, "for_run")(store)
            components = replace(
                components,
                decision_store=store,
                decision_controller=controller,
                validator=validator,
                planner=DecisionPortfolioPlanner(
                    tools=components.tools.catalog(),
                    deterministic_solvers=components.deterministic_solvers,
                    controller=controller,
                ),
            )
        self.config, self.ledger, self.components = config, ledger, components
        self._running = False
        self._has_run = False
        self.workspaces = WorkspaceStore(parent=staging_parent)
        self.engine = JiuwenAdapter(
            ledger=ledger,
            gateway=components.gateway,
            tools=components.tools,
            workspaces=self.workspaces,
            decision_controller=components.decision_controller,
        )
        self.publisher = AnswerOnlyPublisher(self.workspaces)
        self.policy = V1AcceptancePolicy()
        self.progress: dict[str, TaskProgress] = {}

    async def _validate(self, task: TaskSpec, result: AttemptResult) -> ValidationResult:
        assert result.snapshot_sha256 is not None
        record = self.workspaces.records[result.attempt_id]
        context = ToolContext(
            attempt_id=result.attempt_id,
            workspace_root=result.workspace_root,
            source_artifacts=task.artifacts,
            protected_paths=tuple(sorted(record.protected)),
            deadline_monotonic=self._attempt_deadline,
            limits=self.config.attempt_limits.tool_limits,
        )
        temporary = None
        try:
            temporary = await joined_io(self.workspaces.validation_copy, result)
            copied_root = Path(temporary.name).absolute()
            context = replace(context, workspace_root=copied_root)
            copied_result = replace(result, workspace_root=copied_root)
            async with asyncio.timeout_at(self._attempt_deadline):
                validation = await self.components.validator.validate(task, copied_result, context)
            await joined_io(self.workspaces.verify, result)
            return validation
        except asyncio.CancelledError:
            raise
        except Exception:
            return ValidationResult(
                attempt_id=result.attempt_id,
                snapshot_sha256=result.snapshot_sha256,
                checks=(
                    CheckResult(
                        check_id="validator_error",
                        status="inconclusive",
                        blocking=True,
                        message="validation failed",
                    ),
                ),
                status="inconclusive",
            )

        finally:
            if temporary is not None:
                await joined_io(temporary.cleanup)

    async def run(self, tasks_dir: Path) -> RunReport:
        if self._running:
            raise ContractError("overlapping run calls on one AgentRunner are forbidden")
        self._running = True
        try:
            if self.components.decision_store is not None:
                self.components.decision_store.begin_run()
            self.engine.results.clear()
            self.engine.workflow_invocations = 0
            self.engine.component_steps = 0
            if self._has_run:
                self.publisher = AnswerOnlyPublisher(self.workspaces)
            self._has_run = True
            return await self._run(tasks_dir)
        finally:
            self._running = False

    async def _run(self, tasks_dir: Path) -> RunReport:
        validations: list[ValidationResult] = []

        # Initialize the installed SDK before assigning any attempt deadline.
        def prepare_sdk():
            from openjiuwen.core.workflow import Workflow  # type: ignore[import-untyped]

            return Workflow

        await joined_io(prepare_sdk)
        loaded = await DirectoryTaskLoader().load(tasks_dir)
        self.progress = {
            task.task_id: TaskProgress(task_id=task.task_id, status="pending")
            for task in loaded.tasks
        }
        tasks = {task.task_id: task for task in loaded.tasks}
        registry = self.components.registry
        solvers = registry.all_solvers()
        tools = self.components.tools.catalog()
        offers_list = []
        assessment_errors = []
        for task in loaded.tasks:
            for solver in solvers:
                try:
                    offer = solver.assess(task, tools)
                    if offer.task_id != task.task_id or offer.solver_id != solver.solver_id:
                        raise ValueError("invalid solver offer identity")
                    offers_list.append(offer)
                except Exception:
                    # No fabricated offer. BaseException (including cancellation) propagates.
                    assessment_errors.append(
                        AssessmentFailure(
                            task_id=task.task_id,
                            solver_id=solver.solver_id,
                            error=ErrorInfo(
                                code="internal_error", message="solver assessment failed"
                            ),
                        )
                    )
        offers = tuple(offers_list)
        planner = self.components.planner or BaselinePlanner(
            tools=tools, deterministic_solvers=self.components.deterministic_solvers
        )
        receipts: list[SubmissionReceipt] = []
        # A5 starts each attempt from the same immutable task inputs; no solver
        # history is shared across attempts. A repeated generic call after an
        # evidence/validation failure would therefore have no new evidence.
        exhausted_routes: set[tuple[str, str]] = set()
        budget_deferred_tasks: set[str] = set()
        stop_paid = False
        loop = asyncio.get_running_loop()
        try:
            while loop.time() < self.config.deadline_monotonic:
                progress = tuple(self.progress.values())
                now = loop.time()
                available_offers = tuple(
                    offer
                    for offer in offers
                    if (offer.task_id, offer.solver_id) not in exhausted_routes
                    and (not stop_paid or offer.solver_id in self.components.deterministic_solvers)
                )
                decision = planner.choose(
                    loaded.tasks,
                    available_offers,
                    progress,
                    self.ledger.snapshot(),
                    self.config,
                    now,
                )
                if decision is None:
                    wake = next_not_before_wakeup(
                        planner,
                        loaded.tasks,
                        available_offers,
                        progress,
                        self.ledger.snapshot(),
                        self.config,
                        now,
                    )
                    if wake is not None:
                        await asyncio.sleep(
                            max(0, min(wake, self.config.deadline_monotonic) - loop.time())
                        )
                        continue
                    # No running attempts in this sequential driver. Preserve resource uncertainty.
                    for task_id, state in self.progress.items():
                        if state.status in ("pending", "deferred"):
                            self.progress[task_id] = replace(
                                state,
                                status="deferred"
                                if stop_paid
                                or self.ledger.snapshot().blocked
                                or task_id in getattr(planner, "deferred_task_ids", ())
                                or task_id in budget_deferred_tasks
                                else "exhausted",
                            )
                    break
                task, state = tasks[decision.task_id], self.progress[decision.task_id]
                self.progress[task.task_id] = replace(
                    state,
                    status="running",
                    not_before_monotonic=0,
                    attempted_solvers=(*state.attempted_solvers, decision.solver_id),
                )
                self._attempt_deadline = decision.budget.deadline_monotonic
                result = None
                try:
                    result = await self.engine.run_attempt(
                        task, registry.get(decision.solver_id), decision.budget
                    )
                    if (
                        self.components.decision_controller is not None
                        and result.status == "budget_exhausted"
                        and result.error is not None
                        and result.error.code == "budget_exhausted"
                    ):
                        budget_deferred_tasks.add(task.task_id)
                    current = self.progress[task.task_id]
                    current = replace(current, last_result=result)
                    if result.status == "candidate":
                        validation = await self._validate(task, result)
                        if (
                            result.attempt_id
                            in getattr(self.components.gateway, "competition_overruns", ())
                            and self.components.decision_store is not None
                            and not self.components.decision_store.get(task).independent_semantics
                        ):
                            validation = replace(
                                validation,
                                status="failed",
                                checks=(
                                    *validation.checks,
                                    CheckResult(
                                        check_id="overrun_independent_verification",
                                        status="failed",
                                        blocking=True,
                                        message="financial overrun requires independent verification",
                                    ),
                                ),
                            )
                        validations.append(validation)
                        acceptance = self.policy.decide(result, validation)
                        published = False
                        if acceptance.status == "accept":
                            if loop.time() >= decision.budget.deadline_monotonic:
                                stop_paid = True
                            # The owned filesystem operation must finish before
                            # cleanup or another task; a blocked kernel call has no
                            # safely enforceable timeout in a Python worker thread.
                            receipt = await self.publisher.publish(task, result, acceptance)
                            receipts.append(receipt)
                            if loop.time() >= decision.budget.deadline_monotonic:
                                stop_paid = True
                            published = receipt.status in ("saved", "already_saved")
                            current = replace(
                                current,
                                status="submitted"
                                if receipt.status in ("saved", "already_saved")
                                else "exhausted",
                            )
                        else:
                            if (
                                decision.solver_id == "generic"
                                and "generic" not in self.components.deterministic_solvers
                            ):
                                exhausted_routes.add((task.task_id, decision.solver_id))
                            current = replace(current, status="pending")
                        if self.components.decision_store is not None:
                            self.components.decision_store.finish(
                                task, validated=acceptance.status == "accept", published=published
                            )
                    elif result.status in ("internal_error", "tool_error", "cancelled", "deadline"):
                        current = replace(current, status="exhausted")
                    else:
                        error = result.error
                        if (
                            decision.solver_id == "generic"
                            and "generic" not in self.components.deterministic_solvers
                            and (
                                result.status == "blocked_by_evidence"
                                or (
                                    result.status == "provider_error"
                                    and (
                                        (error is not None and error.code == "invalid_output")
                                        or self.ledger.snapshot(result.attempt_id).held_units > 0
                                    )
                                )
                            )
                        ):
                            exhausted_routes.add((task.task_id, decision.solver_id))
                        delay = error.retry_after_s if error is not None else None
                        current = replace(
                            current,
                            status="deferred" if delay else "pending",
                            not_before_monotonic=loop.time() + delay if delay is not None else 0,
                        )
                    if (
                        current.status in ("pending", "deferred")
                        and len(current.attempted_solvers)
                        >= self.config.attempt_limits.max_attempts_per_task
                    ):
                        current = replace(current, status="exhausted")
                    self.progress[task.task_id] = current
                    if task.task_id in budget_deferred_tasks:
                        self.progress[task.task_id] = replace(current, status="deferred")
                except asyncio.CancelledError:
                    last = self.engine.results[-1] if self.engine.results else None
                    saved = self.publisher.receipt_for(task.task_root)
                    if saved is not None:
                        receipts.append(saved)
                    self.progress[task.task_id] = replace(
                        self.progress[task.task_id],
                        status="submitted" if saved is not None else "exhausted",
                        last_result=last
                        if last is not None and last.task_id == task.task_id
                        else result,
                    )
                    raise
                except Exception:
                    # A failed validation/publication must not discard other tasks' receipts.
                    self.progress[task.task_id] = replace(
                        self.progress[task.task_id], status="exhausted", last_result=result
                    )
                finally:
                    store = self.components.decision_store
                    if store is not None:
                        from nsu_budget_agent.solvers.decision import State

                        route = store.get(task)
                        if (
                            task.task_id not in budget_deferred_tasks
                            and decision.solver_id == "generic"
                            and route.state
                            not in (
                                State.SKIP,
                                State.PUBLISH,
                            )
                        ):
                            route.skip("paid attempt ended without publication")
                        # Every route is exhausted after one physical attempt. A
                        # new role/strategy continues inside that bounded workflow.
                        exhausted_routes.add((task.task_id, decision.solver_id))
                    if result is not None:
                        await joined_io(self.workspaces.cleanup, result.attempt_id)
                    elif self.engine.results and self.engine.results[-1].task_id == task.task_id:
                        await joined_io(self.workspaces.cleanup, self.engine.results[-1].attempt_id)
            for task_id, state in self.progress.items():
                if (
                    state.status in ("pending", "deferred")
                    and loop.time() >= self.config.deadline_monotonic
                ):
                    self.progress[task_id] = replace(state, status="exhausted")
            if self.components.decision_store is not None:
                for task in loaded.tasks:
                    route = self.components.decision_store.get(task)
                    if self.progress[task.task_id].status != "deferred":
                        route.skip("portfolio ended without another eligible action")
            return RunReport(
                loaded=loaded,
                progress=tuple(self.progress.values()),
                attempts=tuple(self.engine.results),
                receipts=tuple(receipts),
                budget=self.ledger.snapshot(),
                workflow_invocations=self.engine.workflow_invocations,
                component_steps=self.engine.component_steps,
                assessment_errors=tuple(assessment_errors),
                decision_traces=self.components.decision_store.traces()
                if self.components.decision_store is not None
                else (),
                validations=tuple(validations),
            )
        finally:
            for attempt_id in tuple(self.workspaces.records):
                await joined_io(self.workspaces.cleanup, attempt_id)

    async def aclose(self) -> None:
        try:
            await self.engine.aclose()
        finally:
            # LLMGateway v1 does not require ownership/close. The production
            # composition owns its optional HTTP transport for this run.
            close = getattr(self.components.gateway, "aclose", None)
            if close is not None:
                await close()
