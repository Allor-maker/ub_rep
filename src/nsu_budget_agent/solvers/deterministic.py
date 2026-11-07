"""Stateless production specializations with explicit refusal outside their grammar."""

from nsu_budget_agent.contracts.models import (
    CandidateAnswer,
    SolverOffer,
    SolverStep,
    StepRecord,
    StopAttempt,
    TaskSpec,
    ToolSpec,
    parse_rel_path,
)
from nsu_budget_agent.contracts.ports import SolverContext

from .plans import Unsupported, plan


class DeterministicSolver:
    def __init__(self, kind: str) -> None:
        if kind not in ("math", "search", "data_analysis"):
            raise ValueError("unknown specialization")
        self.kind = kind
        self.solver_id = "local_" + kind

    def assess(self, task: TaskSpec, tools: tuple[ToolSpec, ...]) -> SolverOffer:
        try:
            parsed = plan(task)
            applicable = parsed.kind == self.kind and any(
                t.name == parsed.call.tool_name and t.available for t in tools
            )
        except (Unsupported, ValueError):
            applicable = False
        return SolverOffer(
            task_id=task.task_id,
            solver_id=self.solver_id,
            applicability="yes" if applicable else "no",
            required_tools=(parsed.call.tool_name,) if applicable else (),
            expected_cost_units=0,
            reason="explicit local grammar" if applicable else "no supported local interpretation",
        )

    async def step(self, ctx: SolverContext, history: tuple[StepRecord, ...]) -> SolverStep:
        try:
            parsed = plan(ctx.task)
        except ValueError:
            return StopAttempt(
                status="unsupported", reason="no supported deterministic interpretation"
            )
        if parsed.kind != self.kind:
            return StopAttempt(status="unsupported", reason="wrong specialization")
        if not history:
            return parsed.call
        record = history[-1]
        if record.call != parsed.call or record.result.status != "ok" or record.result.truncated:
            return StopAttempt(
                status="blocked_by_evidence", reason="local evidence could not be established"
            )
        data = record.result.data
        if not isinstance(data, dict):
            return StopAttempt(status="no_progress", reason="invalid tool answer")
        answer, paths = data.get("answer"), data.get("evidence_paths")
        if not isinstance(answer, str) or not isinstance(paths, list):
            return StopAttempt(status="no_progress", reason="invalid tool answer")
        evidence = []
        for path in paths:
            if not isinstance(path, str):
                return StopAttempt(status="no_progress", reason="invalid tool evidence")
            evidence.append(parse_rel_path(path))
        return CandidateAnswer(text=answer, evidence_paths=tuple(evidence))


class UnsupportedGenericSolver:
    solver_id = "generic"

    def assess(self, task: TaskSpec, tools: tuple[ToolSpec, ...]) -> SolverOffer:
        return SolverOffer(
            task_id=task.task_id,
            solver_id=self.solver_id,
            applicability="unknown",
            required_tools=(),
            expected_cost_units=0,
            reason="no LLM fallback installed",
        )

    async def step(self, ctx: SolverContext, history: tuple[StepRecord, ...]) -> SolverStep:
        return StopAttempt(
            status="unsupported",
            reason="task is outside the supported local grammar; LLM fallback disabled",
        )
