"""Real openJiuwen workflows, synthetic B components, no HTTP or model calls."""

import asyncio
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from nsu_budget_agent.budget import BudgetLedger
from nsu_budget_agent.contracts.models import (
    AttemptBudget,
    AttemptLimits,
    ErrorInfo,
    OperationFailed,
    RunConfig,
    ToolLimits,
)
from nsu_budget_agent.orchestration.runtime import AgentRunner

from .fakes import MODEL, FakeGateway, FakeValidator, build


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from nsu_budget_agent.orchestration.sdk_probe import _offline_audit

        sys.addaudithook(_offline_audit)
        # Import the real installed SDK once, outside per-attempt deadlines.
        from openjiuwen.core.workflow import Workflow

        cls.sdk_workflow = Workflow

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tasks = self.root / "tasks"
        self.tasks.mkdir()
        self.staging = self.root / "staging"
        self.staging.mkdir()

    def task(self, name="task", mode="answer", content="  synthetic result\n", workspace=True):
        root = self.tasks / name
        root.mkdir()
        (root / "task.json").write_text(
            json.dumps(
                {
                    "task_id": name,
                    "instruction": mode,
                    "reward": 1,
                    "type": "math",
                    "expected": "SECRET ORACLE",
                }
            ),
            encoding="utf-8",
        )
        work = root / "workspace" if workspace else root
        work.mkdir(exist_ok=True)
        (work / "input.txt").write_bytes(content.encode("utf-8"))
        (root / "expected.txt").write_text("SECRET ORACLE", encoding="utf-8")
        return root

    def runner(self, *, budget=1000, llm=False, validator=None, unknown=False, duration=30):
        now = asyncio.get_running_loop().time()
        config = RunConfig(
            total_budget_units=budget,
            deadline_monotonic=now + 90,
            mode="coverage",
            max_parallel_tasks=1,
            attempt_limits=AttemptLimits(
                max_cost_units=budget,
                max_llm_calls=3,
                max_steps=4,
                max_duration_s=duration,
                max_attempts_per_task=1,
                allowed_models=(MODEL,),
                tool_limits=ToolLimits(
                    timeout_s=2,
                    max_read_bytes=1024,
                    max_output_bytes=1024,
                    max_write_bytes=1024,
                    max_process_memory_bytes=1024,
                    max_process_count=1,
                ),
            ),
        )
        ledger = BudgetLedger(limit_units=budget, deadline_monotonic=config.deadline_monotonic)
        components = build(ledger, config)
        components = dataclasses.replace(
            components,
            gateway=FakeGateway(ledger, unknown=unknown),
            validator=validator or FakeValidator(),
            deterministic_solvers=() if llm else ("generic",),
        )
        runner = AgentRunner(
            config=config, ledger=ledger, components=components, staging_parent=self.staging
        )
        self.addAsyncCleanup(runner.aclose)
        return runner

    async def test_success_exact_bytes_real_workflow_and_isolated_validation(self):
        root = self.task()
        validator = FakeValidator(tamper=True)
        runner = self.runner(validator=validator)
        report = await runner.run(self.tasks)
        self.assertEqual((root / "answer.txt").read_bytes(), b"  synthetic result\n")
        self.assertEqual(report.progress[0].status, "submitted")
        self.assertEqual(report.workflow_invocations, 1)
        self.assertEqual(report.component_steps, 2)
        self.assertEqual(report.attempts[0].steps_used, 2)
        self.assertNotEqual(validator.roots[0], report.attempts[0].workspace_root)
        self.assertEqual(list(self.staging.iterdir()), [])
        self.assertEqual((root / "workspace/input.txt").read_text(), "  synthetic result\n")
        self.assertNotIn("SECRET", (root / "answer.txt").read_text())

    async def test_refusal_error_rejection_empty_and_absence_create_no_answer(self):
        for mode in ("refuse", "unsupported", "error", "reject", "empty", "no_answer"):
            self.task(mode, mode=mode)
        report = await self.runner().run(self.tasks)
        self.assertEqual(len(report.attempts), 6)
        self.assertFalse(any(self.tasks.glob("*/answer.txt")))
        statuses = {a.task_id: a.status for a in report.attempts}
        self.assertEqual(statuses["error"], "internal_error")
        self.assertEqual(statuses["no_answer"], "no_progress")
        self.assertTrue(all(p.status == "exhausted" for p in report.progress))

    async def test_multiple_tasks_preserve_success_on_other_error(self):
        good = self.task("a", workspace=False)
        self.task("b", mode="error")
        self.task("c", content="different")
        report = await self.runner().run(self.tasks)
        self.assertEqual(len(report.receipts), 2)
        self.assertTrue((good / "answer.txt").exists())
        self.assertEqual((self.tasks / "c/answer.txt").read_text(), "different")
        self.assertEqual(len({a.task_id for a in report.attempts}), 3)

    async def test_zero_budget_deterministic_route_still_runs(self):
        self.task()
        report = await self.runner(budget=0).run(self.tasks)
        self.assertEqual(report.progress[0].status, "submitted")
        self.assertEqual(report.budget.spent_units, 0)

    async def test_insufficient_budget_no_llm_or_answer(self):
        root = self.task(mode="llm")
        runner = self.runner(budget=21, llm=True)
        report = await runner.run(self.tasks)
        self.assertFalse((root / "answer.txt").exists())
        self.assertEqual(runner.components.gateway.calls, 0)
        self.assertEqual(report.budget.held_units, 0)

    async def test_llm_usage_recorded_once_by_ledger(self):
        self.task(mode="llm")
        report = await self.runner(llm=True).run(self.tasks)
        self.assertEqual(report.budget.spent_units, 12)
        self.assertEqual(report.budget.held_units, 0)
        self.assertEqual(len(report.attempts[0].usage_request_ids), 1)

    async def test_blocking_inconclusive_does_not_publish(self):
        root = self.task()
        report = await self.runner(validator=FakeValidator(blocking=True)).run(self.tasks)
        self.assertFalse((root / "answer.txt").exists())
        self.assertEqual(report.receipts, ())

    async def test_existing_answer_conflict_never_overwrites(self):
        root = self.task()
        (root / "answer.txt").write_bytes(b"old")
        report = await self.runner().run(self.tasks)
        self.assertEqual(report.receipts[0].status, "conflict")
        self.assertEqual((root / "answer.txt").read_bytes(), b"old")
        self.assertNotEqual(report.progress[0].status, "submitted")

    async def test_cancel_after_dispatch_retains_unknown_reservation(self):
        root = self.task(mode="llm")
        runner = self.runner(llm=True, unknown=True)
        operation = asyncio.create_task(runner.run(self.tasks))
        await asyncio.wait_for(runner.components.gateway.dispatched.wait(), 20)
        operation.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(operation, 10)
        self.assertEqual(runner.ledger.snapshot().held_units, 22)
        self.assertEqual(runner.engine.results[-1].status, "cancelled")
        self.assertFalse((root / "answer.txt").exists())
        self.assertEqual(list(self.staging.iterdir()), [])

    async def test_timeout_after_dispatch_retains_reservation(self):
        root = self.task(mode="llm")
        report = await self.runner(llm=True, unknown=True, duration=0.3).run(self.tasks)
        self.assertEqual(report.attempts[0].status, "deadline")
        self.assertEqual(report.budget.held_units, 22)
        self.assertFalse((root / "answer.txt").exists())

    async def test_subprocess_entrypoint_uses_real_sdk_without_neuraldeep(self):
        root = self.task()
        env = {**os.environ, "BUDGET": "100.5", "PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1"}
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "run_agent.py",
            "--tasks-dir",
            str(self.tasks),
            "--budget-unit",
            "competition_tokens",
            "--components",
            "tests.integration.fakes:build",
            "--staging-dir",
            str(self.staging),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 100)
        self.assertEqual(process.returncode, 0, stderr.decode(errors="replace")[-2000:])
        reports = [
            json.loads(line)
            for line in stdout.decode().splitlines()
            if line.startswith('{"tasks":')
        ]
        self.assertEqual(reports[-1]["saved"], 1)
        self.assertEqual(reports[-1]["workflow_invocations"], 1)
        self.assertEqual((root / "answer.txt").read_bytes(), b"  synthetic result\n")

    async def test_unavailable_explicit_components_fail_closed(self):
        from nsu_budget_agent.orchestration.cli import main

        root = self.task()
        with patch.dict(os.environ, {"BUDGET": "100"}):
            self.assertEqual(
                main(
                    [
                        "--tasks-dir",
                        str(self.tasks),
                        "--budget-unit",
                        "units_x10",
                        "--components",
                        "unavailable_b_package:build",
                    ]
                ),
                2,
            )
        self.assertFalse((root / "answer.txt").exists())

    async def test_deferred_retry_waits_then_publishes(self):
        root = self.task()
        runner = self.runner()
        runner.config = dataclasses.replace(
            runner.config,
            attempt_limits=dataclasses.replace(
                runner.config.attempt_limits, max_attempts_per_task=2
            ),
        )
        solver = runner.components.registry.get("generic")
        original = solver.step
        calls = []

        async def step(ctx, history):
            calls.append(asyncio.get_running_loop().time())
            if len(calls) == 1:
                raise OperationFailed(
                    ErrorInfo(
                        code="rate_limited",
                        message="fixture cooldown",
                        retryable=True,
                        retry_after_s=0.03,
                    )
                )
            return await original(ctx, history)

        solver.step = step
        report = await runner.run(self.tasks)
        self.assertGreaterEqual(calls[1] - calls[0], 0.03)
        self.assertEqual(len(report.attempts), 2)
        self.assertEqual(report.progress[0].status, "submitted")
        self.assertTrue((root / "answer.txt").exists())

    async def test_cancel_during_publication_preserves_committed_status(self):
        root = self.task()
        runner = self.runner()
        started, release = threading.Event(), threading.Event()
        original = runner.publisher._publish

        def delayed(task, result):
            started.set()
            release.wait(5)
            return original(task, result)

        with patch.object(runner.publisher, "_publish", side_effect=delayed):
            operation = asyncio.create_task(runner.run(self.tasks))
            self.assertTrue(await asyncio.to_thread(started.wait, 10))
            operation.cancel()
            await asyncio.sleep(0)
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await operation
        self.assertEqual(runner.progress["task"].status, "submitted")
        self.assertEqual((root / "answer.txt").read_bytes(), b"  synthetic result\n")
        self.assertEqual(list(self.staging.iterdir()), [])

    async def test_tool_identity_and_validator_errors_fail_closed(self):
        root = self.task()
        runner = self.runner()
        original = runner.components.tools.execute

        async def execute(call_id, call, ctx):
            return dataclasses.replace(await original(call_id, call, ctx), call_id="wrong")

        with patch.object(runner.components.tools, "execute", side_effect=execute):
            report = await runner.run(self.tasks)
        self.assertEqual(report.attempts[0].status, "internal_error")
        self.assertFalse((root / "answer.txt").exists())

        another = self.runner()
        with patch.object(
            another.components.validator, "validate", side_effect=RuntimeError("fixture")
        ):
            rejected = await another.run(self.tasks)
        self.assertEqual(rejected.attempts[0].status, "candidate")
        self.assertEqual(rejected.receipts, ())
        self.assertFalse((root / "answer.txt").exists())

    async def test_assess_error_isolated_to_one_task_solver_pair(self):
        failed = self.task("a")
        good = self.task("b", content="second task answer")
        runner = self.runner()
        solver = runner.components.registry.get("generic")
        original = solver.assess

        def assess(task, tools):
            if task.task_id == "a":
                raise RuntimeError("private exception details must not escape")
            return original(task, tools)

        with patch.object(solver, "assess", side_effect=assess):
            report = await runner.run(self.tasks)
        self.assertFalse((failed / "answer.txt").exists())
        self.assertEqual((good / "answer.txt").read_bytes(), b"second task answer")
        self.assertEqual([(a.task_id, a.status) for a in report.attempts], [("b", "candidate")])
        self.assertEqual(len(report.assessment_errors), 1)
        failure = report.assessment_errors[0]
        self.assertEqual((failure.task_id, failure.solver_id), ("a", "generic"))
        self.assertNotIn("private", failure.error.message)
        self.assertEqual(report.progress[0].attempted_solvers, ())

    async def test_assess_does_not_suppress_cancellation_or_system_exceptions(self):
        self.task()
        for error in (asyncio.CancelledError(), KeyboardInterrupt(), SystemExit(1)):
            with self.subTest(error=type(error).__name__):
                runner = self.runner()
                solver = runner.components.registry.get("generic")
                with (
                    patch.object(solver, "assess", side_effect=error),
                    self.assertRaises(type(error)),
                ):
                    await runner.run(self.tasks)
                self.assertEqual(runner.engine.results, [])

    async def test_expired_attempt_open_is_deadline_without_workflow(self):
        from nsu_budget_agent.task_io import OrganizerTaskAdapter

        root = self.task()
        runner = self.runner()
        task = await OrganizerTaskAdapter().load_task(root)
        budget = AttemptBudget(
            max_cost_units=0,
            max_llm_calls=0,
            max_steps=1,
            deadline_monotonic=asyncio.get_running_loop().time() - 0.01,
            allowed_models=(),
            tool_limits=runner.config.attempt_limits.tool_limits,
        )
        result = await runner.engine.run_attempt(
            task, runner.components.registry.get("generic"), budget
        )
        self.assertEqual(result.status, "deadline")
        self.assertEqual(result.error.code, "timeout")
        self.assertEqual(runner.engine.workflow_invocations, 0)
        self.assertEqual(runner.workspaces.records, {})
        self.assertFalse((root / "answer.txt").exists())
