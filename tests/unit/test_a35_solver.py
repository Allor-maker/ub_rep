"""Hand-calculated candidates, independent of LLM output and the B parser."""

import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

from nsu_budget_agent.contracts.models import (
    AttemptBudget,
    CandidateAnswer,
    LLMReply,
    StopAttempt,
    TaskSpec,
    ToolContext,
    ToolLimits,
    UsageRecord,
)
from nsu_budget_agent.contracts.ports import SolverContext
from nsu_budget_agent.solvers.llm import MinimalLLMSolver
from nsu_budget_agent.tools import LocalToolExecutor
from nsu_budget_agent.validation.llm import FallbackValidator, question

MODEL = "qwen3.6-35b-a3b-noreason"
LIMITS = ToolLimits(
    timeout_s=2,
    max_read_bytes=10_000,
    max_output_bytes=10_000,
    max_write_bytes=1000,
    max_process_memory_bytes=10_000_000,
    max_process_count=1,
)


class FallbackFixture(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tools = LocalToolExecutor()

    def task(self, instruction):
        return TaskSpec(
            task_id="task",
            task_root=self.root,
            work_root=self.root,
            instruction=instruction,
            reward=1,
            artifacts=(),
        )

    def context(self, task):
        return ToolContext(
            attempt_id="attempt",
            workspace_root=self.root,
            source_artifacts=task.artifacts,
            protected_paths=(),
            deadline_monotonic=time.monotonic() + 30,
            limits=LIMITS,
        )


class FallbackValidationTests(FallbackFixture):
    async def test_unique_linear_equation_accepts_only_correct_value(self):
        from nsu_budget_agent.contracts.models import AttemptResult
        from nsu_budget_agent.task_io.workspace import snapshot

        # Independent oracle: 3*x+2=14 -> x=4, not any nonempty number.
        task = self.task("Solve for x: 3*x+2=14. Return only the value of x.")
        for answer, expected in (("4", "passed"), ("5", "failed"), ("4.0", "failed")):
            candidate = CandidateAnswer(text=answer)
            result = AttemptResult(
                attempt_id="attempt",
                task_id=task.task_id,
                solver_id="generic",
                status="candidate",
                workspace_root=self.root,
                steps_used=1,
                usage_request_ids=(),
                candidate=candidate,
                snapshot_sha256=snapshot(candidate, task.artifacts),
            )
            checked = await FallbackValidator(self.tools).validate(task, result, self.context(task))
            self.assertEqual(checked.status, expected)

    async def test_unsupported_ambiguity_nonlinearity_and_code_are_refused(self):
        for instruction in (
            "Who will win the election?",
            "Solve for x: x*x=4. Return only the value of x.",
            "Solve for x: x=x. Return only the value of x.",
            "Solve for x: 2*x=1. Return only the value of x. Then double it.",
            "Solve for x: 0*x=3. Return only the value of x.",
            "What is the exact value of 1/3? Return only the result.",
            "What is the exact value of __import__('os')? Return only the result.",
        ):
            self.assertIsNone(question(self.task(instruction)), instruction)
        fractional = self.task("Solve for x: 2*x=1. Return only the value of x.")
        self.assertIsNotNone(question(fractional))
        self.assertIsNone(question(replace(fractional, check_hint="exact_int")))


class GenericSolverTests(FallbackFixture):
    def context_for_solver(self, task, session):
        return SolverContext(
            task=task,
            attempt_id="attempt",
            step_index=0,
            budget=AttemptBudget(
                max_cost_units=2000,
                max_llm_calls=1,
                max_steps=2,
                deadline_monotonic=time.monotonic() + 30,
                allowed_models=(MODEL,),
                tool_limits=LIMITS,
            ),
            tools=self.tools.catalog(),
            llm=session,
        )

    def session(self, text='{"answer":"4","evidence_paths":[]}', *, status="exact", finish="stop"):
        usage = UsageRecord(
            request_id="request",
            attempt_id="attempt",
            model=MODEL,
            status=status,
            input_tokens=20,
            output_tokens=10,
        )
        return AsyncMock(
            complete=AsyncMock(
                return_value=LLMReply(request_id="request", text=text, usage=usage, finish=finish)
            )
        )

    async def test_one_short_call_model_choice_unknown_estimates(self):
        task = self.task("Solve for x: 3*x+2=14. Return only the value of x.")
        solver = MinimalLLMSolver(MODEL)
        offer = solver.assess(task, self.tools.catalog())
        self.assertIsNone(offer.expected_cost_units)
        self.assertIsNone(offer.success_probability)
        session = self.session()
        context = self.context_for_solver(task, session)
        candidate = await solver.step(context, ())
        self.assertEqual(candidate, CandidateAnswer(text="4"))
        request = session.complete.await_args.args[0]
        self.assertEqual(request.model, MODEL)
        self.assertEqual(request.max_output_tokens, 96)
        self.assertIsInstance(await solver.step(replace(context, step_index=1), ()), StopAttempt)
        session.complete.assert_awaited_once()

    async def test_malformed_refusal_unknown_usage_and_truncation(self):
        task = self.task("Solve for x: 3*x+2=14. Return only the value of x.")
        for session in (
            self.session('{"refuse":true}'),
            self.session("not JSON"),
            self.session('{"answer":"4","answer":"5","evidence_paths":[]}'),
            self.session('{"answer":"4","evidence_paths":["expected-answer.txt"]}'),
            self.session(status="unknown"),
            self.session(finish="length"),
        ):
            self.assertIsInstance(
                await MinimalLLMSolver(MODEL).step(self.context_for_solver(task, session), ()),
                StopAttempt,
            )

    async def test_no_semantic_verifier_no_llm_call(self):
        task = self.task("Guess the missing secret.")
        session = self.session()
        self.assertEqual(MinimalLLMSolver(MODEL).assess(task, ()).applicability, "no")
        stopped = await MinimalLLMSolver(MODEL).step(self.context_for_solver(task, session), ())
        self.assertEqual(stopped.status, "blocked_by_evidence")
        session.complete.assert_not_awaited()
