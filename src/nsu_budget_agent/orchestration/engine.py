"""Real openJiuwen workflow: one solver/tool step per SDK component, no retries."""

import asyncio
import importlib.metadata
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .decision import DecisionController

from nsu_budget_agent.contracts.models import (
    AttemptBudget,
    AttemptResult,
    CandidateAnswer,
    ContractError,
    ErrorInfo,
    LLMReply,
    LLMRequest,
    StepRecord,
    StopAttempt,
    TaskSpec,
    ToolCall,
    ToolContext,
)
from nsu_budget_agent.contracts.ports import (
    BudgetController,
    LLMGateway,
    LLMSession,
    Solver,
    SolverContext,
    ToolExecutor,
)
from nsu_budget_agent.task_io.workspace import WorkspaceStore, joined_io


class _StepSession:
    def __init__(self, session: LLMSession, attempt_id: str, ids: list[str]) -> None:
        self._session, self._attempt_id, self._ids = session, attempt_id, ids
        self._used = False

    async def complete(self, request: LLMRequest) -> LLMReply:
        if self._used:
            raise ContractError("one LLM call per step")
        self._used = True
        reply = await self._session.complete(request)
        if reply.usage.attempt_id != self._attempt_id or reply.request_id in self._ids:
            raise ContractError("invalid usage identity")
        self._ids.append(reply.request_id)
        return reply


@dataclass
class _State:
    task: TaskSpec
    solver: Solver
    budget: AttemptBudget
    attempt_id: str
    root: Path
    steps: int = 0
    history: tuple[StepRecord, ...] = ()
    usage_ids: list[str] = field(default_factory=list)
    candidate: CandidateAnswer | None = None
    result: AttemptResult | None = None


class JiuwenAdapter:
    def __init__(
        self,
        *,
        ledger: BudgetController,
        gateway: LLMGateway,
        tools: ToolExecutor,
        workspaces: WorkspaceStore,
        decision_controller: "DecisionController | None" = None,
    ) -> None:
        if importlib.metadata.version("openjiuwen") != "0.1.19":
            raise ContractError("A5-lite requires openjiuwen 0.1.19")
        self.ledger, self.gateway, self.tools, self.workspaces = ledger, gateway, tools, workspaces
        self.decision_controller = decision_controller
        self.results: list[AttemptResult] = []
        self.workflow_invocations = 0
        self.component_steps = 0

    @staticmethod
    def _result(
        state: _State,
        status,
        *,
        reason: str = "",
        error: ErrorInfo | None = None,
        candidate=None,
        changes=(),
        snapshot_sha256=None,
    ) -> AttemptResult:
        return AttemptResult(
            attempt_id=state.attempt_id,
            task_id=state.task.task_id,
            solver_id=state.solver.solver_id,
            status=status,
            workspace_root=state.root,
            steps_used=state.steps,
            usage_request_ids=tuple(state.usage_ids),
            reason=reason,
            error=error,
            candidate=candidate,
            changes=changes,
            snapshot_sha256=snapshot_sha256,
        )

    async def _step(self, state: _State, index: int) -> None:
        if state.result is not None or state.candidate is not None:
            return
        self.component_steps += 1
        state.steps += 1
        in_tool = False
        try:
            session = _StepSession(
                self.gateway.bind(state.attempt_id, index, state.budget),
                state.attempt_id,
                state.usage_ids,
            )
            catalog = self.tools.catalog()
            context = SolverContext(
                task=state.task,
                attempt_id=state.attempt_id,
                step_index=index,
                budget=state.budget,
                tools=catalog,
                llm=session,
            )
            controller = self.decision_controller
            if controller is not None and state.solver.solver_id == "local_router":
                action = await controller.free.step(context, state.history)
            elif controller is not None and state.solver.solver_id == "generic":
                internal_action = await controller.advance(context, state.history)
                if internal_action is None:
                    return  # A5 role transition consumes this SDK step, not a SolverStep.
                action = internal_action
            else:
                action = await state.solver.step(context, state.history)
            if isinstance(action, CandidateAnswer):
                if len(action.text.encode("utf-8")) > state.budget.tool_limits.max_output_bytes:
                    state.result = self._result(
                        state, "no_progress", reason="candidate exceeds output limit"
                    )
                else:
                    state.candidate = action
            elif isinstance(action, StopAttempt):
                state.result = self._result(state, action.status, reason=action.reason)
            elif isinstance(action, ToolCall):
                specs = {tool.name: tool for tool in catalog}
                if action.tool_name not in specs or not specs[action.tool_name].available:
                    state.result = self._result(
                        state, "unsupported", reason="required tool unavailable"
                    )
                    return
                in_tool = True
                record = self.workspaces.records[state.attempt_id]
                tool_context = ToolContext(
                    attempt_id=state.attempt_id,
                    workspace_root=state.root,
                    source_artifacts=state.task.artifacts,
                    protected_paths=tuple(sorted(record.protected)),
                    deadline_monotonic=state.budget.deadline_monotonic,
                    limits=state.budget.tool_limits,
                )
                call_id = uuid4().hex
                async with asyncio.timeout(
                    min(
                        state.budget.tool_limits.timeout_s,
                        max(0, state.budget.deadline_monotonic - asyncio.get_running_loop().time()),
                    )
                ):
                    result = await self.tools.execute(call_id, action, tool_context)
                if result.call_id != call_id or result.tool_name != action.tool_name:
                    raise ContractError("tool result identity mismatch")
                await joined_io(self.workspaces.protect, state.attempt_id, result.protected_paths)
                state.history = (
                    *state.history,
                    StepRecord(step_index=index, call=action, result=result),
                )
            else:
                raise ContractError("invalid SolverStep")
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            expired = asyncio.get_running_loop().time() >= state.budget.deadline_monotonic
            error = ErrorInfo(code="timeout", message="step timed out", retryable=True)
            state.result = self._result(
                state,
                "deadline" if expired else ("tool_error" if in_tool else "provider_error"),
                reason="step timed out",
                error=error,
            )
        except Exception as exc:
            from nsu_budget_agent.contracts.models import OperationFailed

            if isinstance(exc, OperationFailed):
                error = exc.error
                status = "tool_error" if in_tool else "provider_error"
                if error.code in ("budget_exhausted", "quota_exhausted"):
                    status = "budget_exhausted"
                    if error.code == "quota_exhausted":
                        self.ledger.halt(error)
                elif error.code == "unsupported":
                    status = "unsupported"
                elif (
                    error.code == "timeout"
                    and asyncio.get_running_loop().time() >= state.budget.deadline_monotonic
                ):
                    status = "deadline"
                state.result = self._result(
                    state, status, reason="step service refused", error=error
                )
            else:
                state.result = self._result(
                    state,
                    "internal_error",
                    reason="step failed",
                    error=ErrorInfo(code="internal_error", message="step failed"),
                )

    async def run_attempt(
        self, task: TaskSpec, solver: Solver, budget: AttemptBudget
    ) -> AttemptResult:
        from openjiuwen.core.workflow import (  # type: ignore[import-untyped]
            End,
            Start,
            Workflow,
            WorkflowCard,
            WorkflowComponent,
            WorkflowExecutionState,
            create_workflow_session,
        )

        attempt_id = uuid4().hex
        state = _State(
            task, solver, budget, attempt_id, Path.cwd() / "uncreated-staging" / attempt_id
        )
        opened = False
        adapter = self

        class Step(WorkflowComponent):
            def __init__(self, index: int):
                super().__init__()
                self.index = index

            async def invoke(self, inputs, session, context):
                await adapter._step(state, self.index)
                return {"tick": self.index + 1}

        try:
            self.ledger.open_attempt(attempt_id, budget)
            opened = True
            state.root = await joined_io(self.workspaces.create, task, attempt_id)
            async with asyncio.timeout_at(budget.deadline_monotonic):
                flow = Workflow(
                    card=WorkflowCard(id=attempt_id, name="budget-attempt", version="1")
                )
                flow.set_start_comp("start", Start(), inputs_schema={"tick": "${tick}"})
                previous = "start"
                for index in range(budget.max_steps):
                    current = f"step{index}"
                    flow.add_workflow_comp(
                        current,
                        Step(index),
                        max_retries=0,
                        timeout=-1.0,
                        inputs_schema={"tick": "${" + previous + ".tick}"},
                    )
                    flow.add_connection(previous, current)
                    previous = current
                flow.set_end_comp("end", End(), inputs_schema={"tick": "${" + previous + ".tick}"})
                flow.add_connection(previous, "end")
                self.workflow_invocations += 1
                output = await flow.invoke({"tick": 0}, create_workflow_session())
                if output.state != WorkflowExecutionState.COMPLETED:
                    raise RuntimeError("workflow did not complete")
                if state.candidate is not None:
                    changes, sha = await joined_io(
                        self.workspaces.freeze, attempt_id, state.candidate
                    )
                    state.result = self._result(
                        state,
                        "candidate",
                        candidate=state.candidate,
                        changes=changes,
                        snapshot_sha256=sha,
                    )
                elif state.result is None:
                    state.result = self._result(state, "no_progress", reason="step limit reached")
        except asyncio.CancelledError:
            state.result = self._result(state, "cancelled", reason="attempt cancelled")
            raise
        except TimeoutError:
            state.result = self._result(state, "deadline", reason="attempt deadline expired")
        except Exception as exc:
            from nsu_budget_agent.contracts.models import OperationFailed

            error = (
                exc.error
                if isinstance(exc, OperationFailed)
                else ErrorInfo(
                    code="internal_error", message="attempt setup or finalization failed"
                )
            )
            if (
                error.code == "timeout"
                and asyncio.get_running_loop().time() >= budget.deadline_monotonic
            ):
                status = "deadline"
            elif error.code in ("budget_exhausted", "quota_exhausted"):
                status = "budget_exhausted"
            else:
                status = "tool_error" if isinstance(exc, OperationFailed) else "internal_error"
            state.result = self._result(state, status, reason="attempt failed", error=error)
        finally:
            if opened:
                self.ledger.close_attempt(attempt_id)
            if state.result is not None:
                self.results.append(state.result)
        assert state.result is not None
        return state.result

    async def aclose(self) -> None:
        from openjiuwen.core.runner import Runner  # type: ignore[import-untyped]

        await Runner.stop()
