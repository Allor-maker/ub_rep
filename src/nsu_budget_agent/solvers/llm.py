"""One completion: a checked local data plan or an independently checked number."""

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
    ToolCall,
    ToolSpec,
    parse_rel_path,
)
from nsu_budget_agent.contracts.ports import SolverContext
from nsu_budget_agent.validation.data_plan import data_question, evidence_fits, matches_instruction
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

    def __init__(
        self, model: ModelId, *, max_output_tokens: int = 96, max_plan_output_tokens: int = 192
    ) -> None:
        # DTO validates both model and cap without any network operation.
        LLMRequest(
            model=model,
            messages=(ChatMessage(role="user", content=""),),
            max_output_tokens=max_output_tokens,
        )
        self.model, self.max_output_tokens = model, max_output_tokens
        LLMRequest(
            model=model,
            messages=(ChatMessage(role="user", content=""),),
            max_output_tokens=max_plan_output_tokens,
        )
        self.max_plan_output_tokens = max_plan_output_tokens

    def assess(self, task: TaskSpec, tools: tuple[ToolSpec, ...]) -> SolverOffer:
        spec = data_question(task)
        required = (spec.call.tool_name, "read_file") if spec is not None else ()
        supported = (question(task) is not None or spec is not None) and set(required).issubset(
            t.name for t in tools if t.available and not t.mutates_files
        )
        return SolverOffer(
            task_id=task.task_id,
            solver_id=self.solver_id,
            applicability="yes" if supported else "no",
            required_tools=required,
            reason="independent semantic verifier available"
            if supported
            else "no sufficient semantic verifier",
        )

    async def step(self, ctx: SolverContext, history: tuple[StepRecord, ...]) -> SolverStep:
        if data_question(ctx.task) is not None:
            return await self._data_step(ctx, history)
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

    async def _data_step(self, ctx: SolverContext, history: tuple[StepRecord, ...]) -> SolverStep:
        spec = data_question(ctx.task)
        assert spec is not None
        if ctx.step_index == 1 and len(history) == 1 and history[0].step_index == 0:
            record = history[0]
            data = record.result.data
            answer = data.get("answer") if isinstance(data, dict) else None
            evidence = data.get("evidence_paths") if isinstance(data, dict) else None
            if (
                not matches_instruction(ctx.task, record.call)
                or record.result.status != "ok"
                or record.result.truncated
                or not isinstance(data, dict)
                or not isinstance(answer, str)
                or not answer
                or not isinstance(evidence, list)
            ):
                return StopAttempt(
                    status="blocked_by_evidence", reason="local plan did not establish an answer"
                )
            try:
                paths_list = []
                for raw_path in evidence:
                    if not isinstance(raw_path, str):
                        raise ValueError("invalid evidence path")
                    paths_list.append(parse_rel_path(raw_path))
                paths = tuple(paths_list)
                if not paths or not set(paths).issubset(spec.paths):
                    raise ValueError("invalid evidence")
                return CandidateAnswer(text=answer, evidence_paths=paths)
            except (ValueError, TypeError):
                return StopAttempt(status="blocked_by_evidence", reason="invalid local evidence")
        if ctx.step_index != 0 or history:
            return StopAttempt(status="no_progress", reason="one plan and one local execution only")
        if (
            self.model not in ctx.budget.allowed_models
            or ctx.budget.max_llm_calls < 1
            or ctx.budget.max_steps < 2
        ):
            return StopAttempt(status="unsupported", reason="plan budget unavailable")
        if not self.assess(ctx.task, ctx.tools).applicability == "yes":
            return StopAttempt(status="unsupported", reason="required read-only tools unavailable")
        if not evidence_fits(ctx.task, spec, ctx.budget.tool_limits):
            return StopAttempt(
                status="blocked_by_evidence", reason="evidence exceeds read or validation limits"
            )
        reply = await ctx.llm.complete(
            LLMRequest(
                model=self.model,
                messages=(
                    ChatMessage(
                        role="system",
                        content=(
                            "Translate the instruction into exactly one JSON action: "
                            '{"kind":"tool","tool_name":"query_csv or lookup_markdown","arguments":{...}}. '
                            "Never compute or include the final answer, code, or extra keys. "
                            "query_csv arguments: path, aggregate (sum, count_rows, product_sum), "
                            "columns (zero, one or two exact column names), filters "
                            '([{"column":"literal column","op":"eq","value":"literal value"}]). '
                            "lookup_markdown arguments: paths (all listed Markdown paths), entity, "
                            'chain ([] or ["Employer"]), field (Email, Department, Headquarters, Born, Founded). '
                            'Use only the supplied artifact paths. Return {"kind":"stop","status":"blocked_by_evidence","reason":"unsupported"} if ambiguous.'
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=json.dumps(
                            {
                                "instruction": ctx.task.instruction,
                                "artifact_paths": [p.as_posix() for p in spec.paths],
                            },
                            ensure_ascii=True,
                        ),
                    ),
                ),
                max_output_tokens=self.max_plan_output_tokens,
            )
        )
        if (
            reply.finish != "stop"
            or reply.usage.status != "exact"
            or len(reply.text.encode("utf-8")) > 4096
        ):
            return StopAttempt(
                status="blocked_by_evidence", reason="incomplete plan or unknown usage"
            )
        try:
            action = json.loads(reply.text, object_pairs_hook=_unique_object)
            if (
                not isinstance(action, dict)
                or set(action) != {"kind", "tool_name", "arguments"}
                or action["kind"] != "tool"
            ):
                raise ValueError("invalid plan envelope")
            call = ToolCall(tool_name=action["tool_name"], arguments=action["arguments"])
            if not matches_instruction(ctx.task, call):
                raise ValueError("plan does not match instruction semantics")
            return call
        except (ValueError, TypeError, RecursionError):
            return StopAttempt(
                status="blocked_by_evidence",
                reason="plan rejected by independent instruction check",
            )
