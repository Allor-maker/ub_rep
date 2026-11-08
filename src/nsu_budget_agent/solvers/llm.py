"""One short text completion, no model-directed tools or internal agent loop."""

import json

from nsu_budget_agent.contracts.models import (
    CandidateAnswer,
    ChatMessage,
    LLMRequest,
    ModelId,
    SolverOffer,
    SolverStep,
    StepRecord,
    StopAttempt,
    TaskSpec,
    ToolSpec,
)
from nsu_budget_agent.contracts.ports import SolverContext
from nsu_budget_agent.validation.llm import question


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate response key")
        result[key] = value
    return result


class MinimalLLMSolver:
    solver_id = "generic"

    def __init__(self, model: ModelId, *, max_output_tokens: int = 96) -> None:
        # DTO validates both model and cap without any network operation.
        LLMRequest(
            model=model,
            messages=(ChatMessage(role="user", content=""),),
            max_output_tokens=max_output_tokens,
        )
        self.model, self.max_output_tokens = model, max_output_tokens

    def assess(self, task: TaskSpec, tools: tuple[ToolSpec, ...]) -> SolverOffer:
        supported = question(task) is not None
        return SolverOffer(
            task_id=task.task_id,
            solver_id=self.solver_id,
            applicability="yes" if supported else "no",
            required_tools=(),
            reason="independent semantic verifier available"
            if supported
            else "no sufficient semantic verifier",
        )

    async def step(self, ctx: SolverContext, history: tuple[StepRecord, ...]) -> SolverStep:
        if ctx.step_index != 0 or history:
            return StopAttempt(status="no_progress", reason="single completion only")
        if question(ctx.task) is None:
            return StopAttempt(
                status="blocked_by_evidence", reason="no sufficient semantic verifier"
            )
        if self.model not in ctx.budget.allowed_models or ctx.budget.max_llm_calls < 1:
            return StopAttempt(status="unsupported", reason="LLM capability unavailable")
        reply = await ctx.llm.complete(
            LLMRequest(
                model=self.model,
                messages=(
                    ChatMessage(
                        role="system",
                        content='Return only JSON {"answer":"exact number","evidence_paths":[]}. '
                        'No explanation, markdown, tools or reasoning text. If uncertain, return {"refuse":true}.',
                    ),
                    ChatMessage(role="user", content=ctx.task.instruction),
                ),
                max_output_tokens=self.max_output_tokens,
            )
        )
        if (
            reply.finish != "stop"
            or reply.usage.status != "exact"
            or len(reply.text.encode("utf-8")) > 4096
        ):
            return StopAttempt(
                status="blocked_by_evidence", reason="incomplete response or unknown usage"
            )
        try:
            data = json.loads(reply.text, object_pairs_hook=_unique_object)
            if (
                not isinstance(data, dict)
                or set(data) != {"answer", "evidence_paths"}
                or not isinstance(data["answer"], str)
                or not data["answer"]
                or data["evidence_paths"] != []
            ):
                raise ValueError("invalid response")
            return CandidateAnswer(text=data["answer"])
        except (ValueError, TypeError, RecursionError):
            return StopAttempt(
                status="blocked_by_evidence", reason="model did not provide a valid candidate"
            )
