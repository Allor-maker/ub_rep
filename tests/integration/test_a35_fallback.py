"""Production composition + real SDK + real A2/A3 with fake HTTP only."""

import asyncio
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
from tests.unit.test_a35_provider import ENV, MODELS, mock_compatibility

from nsu_budget_agent.budget import BudgetLedger
from nsu_budget_agent.contracts.models import AttemptLimits, ContractError, RunConfig, ToolLimits
from nsu_budget_agent.orchestration.runtime import AgentRunner
from nsu_budget_agent.solvers.composition import build, build_with_llm


class FallbackPipelineTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from openjiuwen.core.workflow import Workflow

        cls.sdk = Workflow

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tasks = self.root / "tasks"
        self.staging = self.root / "staging"
        self.tasks.mkdir()
        self.staging.mkdir()
        self.calls = []
        guard = patch.object(
            httpx.AsyncHTTPTransport,
            "handle_async_request",
            side_effect=AssertionError("real NeuralDeep forbidden"),
        )
        guard.start()
        self.addCleanup(guard.stop)

    def add_task(
        self, name="equation", instruction="Solve for x: 3*x+2=14. Return only the value of x."
    ):
        root = self.tasks / name
        root.mkdir()
        (root / "task.json").write_text(
            json.dumps(
                {
                    "task_id": name,
                    "instruction": instruction,
                    "reward": 1,
                    "type": "math",
                    "check": "exact_int",
                }
            ),
            encoding="utf-8",
        )
        return root

    def config(self, limit):
        return RunConfig(
            total_budget_units=limit,
            deadline_monotonic=asyncio.get_running_loop().time() + 120,
            attempt_limits=AttemptLimits(
                max_cost_units=limit,
                max_llm_calls=1,
                max_steps=2,
                max_duration_s=20,
                max_attempts_per_task=1,
                allowed_models=MODELS,
                tool_limits=ToolLimits(
                    timeout_s=2,
                    max_read_bytes=100_000,
                    max_output_bytes=100_000,
                    max_write_bytes=1000,
                    max_process_memory_bytes=10_000_000,
                    max_process_count=1,
                ),
            ),
        )

    def runner(
        self,
        *,
        answer="4",
        usage=True,
        limit=5000,
        model=MODELS[0],
        timeout=False,
        http_status=200,
        malformed_choice=False,
        attempts=1,
        mode="auto",
    ):
        config = self.config(limit)
        config = replace(
            config,
            mode=mode,
            attempt_limits=replace(config.attempt_limits, max_attempts_per_task=attempts),
        )
        ledger = BudgetLedger(limit_units=limit, deadline_monotonic=config.deadline_monotonic)

        def handler(request):
            self.calls.append(json.loads(request.content))
            self.assertGreater(ledger.snapshot().held_units, 0)
            if timeout:
                raise httpx.ReadTimeout("private provider failure")
            if http_status != 200:
                return httpx.Response(http_status, json={"error": {"code": "temporary"}})
            data = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"answer": answer, "evidence_paths": []})
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
            if usage:
                data["usage"] = {
                    "prompt_tokens": 20,
                    "completion_tokens": 12,
                    "completion_tokens_details": {"reasoning_tokens": 2},
                }
            if malformed_choice:
                data["choices"] = []
            return httpx.Response(200, json=data)

        components = build_with_llm(
            ledger,
            config,
            compatibility=mock_compatibility(),
            transport=httpx.MockTransport(handler),
            environ={**ENV, "LLM_MODEL": model},
        )
        runner = AgentRunner(
            config=config, ledger=ledger, components=components, staging_parent=self.staging
        )
        self.addAsyncCleanup(runner.aclose)
        return runner

    async def test_four_model_routes_answer_only_after_validation(self):
        for index, model in enumerate(MODELS):
            root = self.add_task(str(index))
            runner = self.runner(model=model)
            report = await runner.run(self.tasks)
            self.assertEqual((root / "answer.txt").read_bytes(), b"4")
            self.assertEqual(self.calls[-1]["model"], model)
            self.assertEqual(report.budget.spent_units, 140)
            self.assertEqual(report.budget.held_units, 0)
            self.assertGreater(report.workflow_invocations, 0)
            # Remove only our published fixture, so the next route tests a fresh
            # task directory rather than relying on old answer discovery.
            (root / "answer.txt").unlink()
            (root / "task.json").unlink()
            root.rmdir()

    async def test_wrong_answer_is_not_published(self):
        root = self.add_task()
        report = await self.runner(answer="5").run(self.tasks)
        self.assertFalse((root / "answer.txt").exists())
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(report.budget.spent_units, 140)
        self.assertEqual(report.receipts, ())

    async def test_insufficient_budget_no_dispatch(self):
        root = self.add_task()
        report = await self.runner(limit=100).run(self.tasks)
        self.assertEqual(self.calls, [])
        self.assertEqual(report.budget.spent_units, 0)
        self.assertEqual(report.budget.held_units, 0)
        self.assertFalse((root / "answer.txt").exists())

    async def test_unknown_usage_and_timeout_retain_reserve_no_publication(self):
        root = self.add_task()
        for options in (
            {"usage": False},
            {"timeout": True},
            {"http_status": 429},
            {"http_status": 500},
        ):
            runner = self.runner(**options)
            report = await runner.run(self.tasks)
            self.assertEqual(report.budget.spent_units, 0)
            self.assertEqual(report.budget.held_units, 128 + 10 * 96)
            self.assertFalse((root / "answer.txt").exists())

    async def test_rejected_generic_candidate_is_not_resent(self):
        root = self.add_task()
        report = await self.runner(answer="5", attempts=2).run(self.tasks)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(report.budget.spent_units, 140)
        self.assertEqual(report.budget.held_units, 0)
        self.assertFalse((root / "answer.txt").exists())

    async def test_invalid_provider_output_with_exact_usage_is_not_resent(self):
        root = self.add_task()
        report = await self.runner(malformed_choice=True, attempts=2).run(self.tasks)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(report.attempts[0].error.code, "invalid_output")
        self.assertEqual(report.budget.spent_units, 140)
        self.assertEqual(report.budget.held_units, 0)
        self.assertFalse((root / "answer.txt").exists())

    async def test_unknown_or_timed_out_generic_is_not_resent(self):
        root = self.add_task()
        for options in (
            {"usage": False},
            {"timeout": True},
            {"http_status": 429},
            {"http_status": 500},
        ):
            self.calls.clear()
            report = await self.runner(attempts=2, **options).run(self.tasks)
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(report.budget.spent_units, 0)
            self.assertEqual(report.budget.held_units, 1088)
            self.assertFalse((root / "answer.txt").exists())

    async def test_coverage_tries_other_tasks_and_different_solver(self):
        from nsu_budget_agent.contracts.models import SolverOffer, StopAttempt

        class Alternative:
            solver_id = "alternative"

            def assess(self, task, tools):
                return SolverOffer(
                    task_id=task.task_id,
                    solver_id=self.solver_id,
                    applicability="unknown" if task.task_id == "first" else "no",
                    required_tools=(),
                    reason="alternative approach",
                )

            async def step(self, ctx, history):
                return StopAttempt(status="unsupported", reason="no result")

        self.add_task("first")
        self.add_task("second")
        runner = self.runner(answer="5", attempts=2, mode="coverage")
        runner.components.registry.register(Alternative())
        report = await runner.run(self.tasks)
        self.assertEqual([attempt.task_id for attempt in report.attempts[:2]], ["first", "second"])
        self.assertEqual(len(self.calls), 2)
        self.assertIn(
            ("first", "alternative"),
            [(attempt.task_id, attempt.solver_id) for attempt in report.attempts],
        )

    async def test_deterministic_first_and_unverifiable_task_no_api(self):
        local = self.add_task("local", "Compute 2+3.")
        unknown = self.add_task("unknown", "What is the meaning of life?")
        report = await self.runner().run(self.tasks)
        self.assertEqual((local / "answer.txt").read_bytes(), b"5")
        self.assertFalse((unknown / "answer.txt").exists())
        self.assertEqual(self.calls, [])
        self.assertEqual(report.budget.spent_units, 0)

    async def test_two_tasks_share_real_ledger(self):
        self.add_task("a")
        self.add_task("b")
        report = await self.runner(limit=1250).run(self.tasks)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(report.budget.spent_units, 280)
        self.assertEqual(report.budget.held_units, 0)
        self.assertEqual(sum(r.status == "saved" for r in report.receipts), 2)

    async def test_production_default_offline_and_explicit_factory_fail_closed(self):
        config = self.config(5000)
        ledger = BudgetLedger(limit_units=5000, deadline_monotonic=config.deadline_monotonic)
        with patch.dict(os.environ, ENV, clear=True):
            self.assertEqual(type(build(ledger, config).gateway).__name__, "DisabledLLMGateway")
            with self.assertRaises(ContractError):
                build_with_llm(ledger, config)

    async def test_runner_closes_gateway_transport(self):
        runner = self.runner()
        await runner.aclose()
        self.assertTrue(runner.components.gateway._closed)

    async def test_production_rejects_best_effort_and_unconfirmed_profiles(self):
        from nsu_budget_agent.contracts.models import ProviderProfile

        config = self.config(5000)
        ledger = BudgetLedger(limit_units=5000, deadline_monotonic=config.deadline_monotonic)
        with self.assertRaises(ContractError):
            build_with_llm(
                ledger,
                replace(config, accounting_mode="best_effort"),
                compatibility=mock_compatibility(),
                environ=ENV,
            )
        with self.assertRaises(ContractError):
            build_with_llm(
                ledger,
                config,
                compatibility=replace(
                    mock_compatibility(), profiles={m: ProviderProfile() for m in MODELS}
                ),
                environ=ENV,
            )
