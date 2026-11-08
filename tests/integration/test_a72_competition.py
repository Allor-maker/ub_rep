"""Real SDK/A2/A3/A4/A5 competition integration, with network-denying mocks."""

import asyncio
import json
import os
import sys
import threading
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

import httpx
from tests.integration import test_a71_router as harness
from tests.integration.test_a71_router import direct, review
from tests.unit.test_a35_provider import ENV, MODELS

from nsu_budget_agent.budget import BudgetLedger
from nsu_budget_agent.contracts.models import AttemptLimits, ContractError, RunConfig, ToolLimits
from nsu_budget_agent.orchestration.runtime import AgentRunner
from nsu_budget_agent.solvers import composition


class CompetitionPipelineTests(unittest.IsolatedAsyncioTestCase):
    setUpClass = classmethod(harness.RouterPipelineTests.setUpClass.__func__)
    setUp = harness.RouterPipelineTests.setUp
    task = harness.RouterPipelineTests.task

    async def test_blocked_fsync_keeps_staging_and_publishes_once_before_next_task(self):
        first = self.task("first", "Compute 2+3.", reward=100)
        self.task("second", "Give the result of adding eleven and four.", reward=1)
        entered, release = threading.Event(), threading.Event()
        original_fsync = os.fsync

        def blocked_fsync(descriptor):
            if not entered.is_set():
                entered.set()
                if not release.wait(20):
                    raise TimeoutError("test synchronization")
            return original_fsync(descriptor)

        def handler(*_):
            self.assertTrue(release.is_set())
            self.assertEqual((first / "answer.txt").read_bytes(), b"5")
            return httpx.Response(429, json={"error": {"code": "rate_limit"}})

        runner = self.runner(handler=handler)
        with (
            patch("nsu_budget_agent.submission.publisher.os.fsync", side_effect=blocked_fsync),
            patch("nsu_budget_agent.submission.publisher.os.link", wraps=os.link) as link,
            patch.object(runner.workspaces, "cleanup", wraps=runner.workspaces.cleanup) as cleanup,
        ):
            run = asyncio.create_task(runner.run(self.tasks))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 10))
                await asyncio.sleep(5.2)  # Beyond the former ineffective publication timeout.
                self.assertFalse(run.done())
                self.assertEqual(self.calls, [])
                self.assertIsNone(runner.publisher.receipt_for(first))
                self.assertFalse((first / "answer.txt").exists())
                self.assertEqual(len(list(first.glob(".nsu-answer-*.tmp"))), 1)
                self.assertEqual(len(runner.workspaces.records), 1)
                staging = next(iter(runner.workspaces.records.values())).temporary.name
                self.assertTrue(os.path.isdir(staging))
                cleanup.assert_not_called()
            finally:
                release.set()
            report = await asyncio.wait_for(run, 15)
        self.assertEqual((first / "answer.txt").read_bytes(), b"5")
        self.assertEqual([r.status for r in report.receipts], ["saved"])
        self.assertEqual(link.call_count, 1)
        self.assertFalse(os.path.exists(staging))
        self.assertEqual(list(first.glob(".nsu-answer-*.tmp")), [])
        self.assertEqual(len(self.calls), 1)

    async def test_delayed_fsync_failure_returns_real_failed_receipt(self):
        first = self.task("first", "Compute 2+3.")
        entered, release = threading.Event(), threading.Event()

        def failed_fsync(_descriptor):
            entered.set()
            if not release.wait(20):
                raise TimeoutError("test synchronization")
            raise OSError("simulated filesystem failure")

        runner = self.runner()
        with patch("nsu_budget_agent.submission.publisher.os.fsync", side_effect=failed_fsync):
            run = asyncio.create_task(runner.run(self.tasks))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 10))
                await asyncio.sleep(5.2)
                self.assertFalse(run.done())
                self.assertIsNone(runner.publisher.receipt_for(first))
                self.assertEqual(len(runner.workspaces.records), 1)
            finally:
                release.set()
            report = await asyncio.wait_for(run, 15)
        self.assertEqual([r.status for r in report.receipts], ["failed"])
        self.assertFalse((first / "answer.txt").exists())
        self.assertEqual(list(first.glob(".nsu-answer-*.tmp")), [])

    async def test_excess_reported_usage_keeps_independently_verifiable_answer(self):
        for prompt, completion, limit in (
            (40000, 48, 100000),
            (12000, 48, 10000),
            (120, 600, 100000),
        ):
            with self.subTest(prompt=prompt, completion=completion, limit=limit):
                self.tasks = self.root / f"case-{prompt}"
                self.tasks.mkdir()
                root = self.task("affine", "Solve for x: 3*x+2=14. Return only the value of x.")
                self.task("other", "Give the result of adding eleven and four.", reward=1)
                runner = self.runner(
                    limit=limit,
                    usage={
                        "prompt_tokens": prompt,
                        "completion_tokens": completion,
                        "total_tokens": prompt + completion,
                        "completion_tokens_details": {"reasoning_tokens": 0},
                    },
                    handler=lambda *_: direct("4"),
                )
                report = await runner.run(self.tasks)
                reserved = runner.components.gateway.events[0]
                self.assertTrue(
                    prompt > reserved.input_bound_tokens or completion > reserved.output_cap_tokens,
                    "fixture must exceed the actual current prompt estimate or output cap",
                )
                self.assertEqual(len(self.calls), 1)
                self.assertEqual(report.budget.spent_units, prompt + 10 * completion)
                self.assertEqual(report.budget.held_units, 0)
                self.assertTrue(report.budget.blocked)
                self.assertEqual((root / "answer.txt").read_text(), "4")
                self.assertFalse((self.tasks / "other" / "answer.txt").exists())
                self.calls.clear()

    async def test_excess_usage_does_not_bypass_mandatory_review(self):
        root = self.task("unknown", "Give the result of adding eleven and four.")
        runner = self.runner(
            usage={
                "prompt_tokens": 40000,
                "completion_tokens": 48,
                "total_tokens": 40048,
                "completion_tokens_details": {"reasoning_tokens": 0},
            }
        )
        report = await runner.run(self.tasks)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(report.budget.spent_units, 40480)
        self.assertTrue(report.budget.blocked)
        self.assertFalse((root / "answer.txt").exists())

    async def test_overrun_in_reviewer_does_not_certify_unknown_semantics(self):
        root = self.task("unknown", "Give the result of adding eleven and four.")

        def handler(wire, payload):
            if payload["role"] == "solve":
                return direct("15")
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": json.dumps(review())}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 9000,
                        "completion_tokens": 48,
                        "total_tokens": 9048,
                        "completion_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            )

        report = await self.runner(handler=handler).run(self.tasks)
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(report.budget.blocked)
        self.assertFalse((root / "answer.txt").exists())

    async def test_overrun_still_rejects_wrong_and_incomplete_answers(self):
        for answer, finish in (("5", "stop"), ("4", "length")):
            self.tasks = self.root / f"bad-{finish}"
            self.tasks.mkdir()
            root = self.task("affine", "Solve for x: 3*x+2=14. Return only the value of x.")

            def handler(*_):
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {"content": json.dumps(direct(answer))},
                                "finish_reason": finish,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 40000,
                            "completion_tokens": 48,
                            "total_tokens": 40048,
                            "completion_tokens_details": {"reasoning_tokens": 0},
                        },
                    },
                )

            report = await self.runner(handler=handler).run(self.tasks)
            self.assertTrue(report.budget.blocked)
            self.assertFalse((root / "answer.txt").exists())

    async def test_accepted_answer_publishes_after_attempt_deadline_without_paid_followup(self):
        root = self.task("free", "Compute 2+3.", reward=90)
        self.task("paid", "Give the result of adding eleven and four.", reward=50)
        runner = self.runner()
        runner.config = replace(
            runner.config, attempt_limits=replace(runner.config.attempt_limits, max_duration_s=2.0)
        )
        decide = runner.policy.decide

        def expire(result, validation):
            decision = decide(result, validation)
            if decision.status == "accept":
                time.sleep(
                    max(0, runner._attempt_deadline - asyncio.get_running_loop().time()) + 0.03
                )
            return decision

        with patch.object(runner.policy, "decide", side_effect=expire):
            report = await runner.run(self.tasks)
        self.assertEqual((root / "answer.txt").read_text(), "5")
        self.assertEqual(len(report.receipts), 1)
        self.assertEqual(self.calls, [])

    async def test_preflight_negotiates_caps_for_mandatory_review(self):
        root = self.task("high", "Give the result of adding eleven and four.", reward=90)
        report = await self.runner(limit=10000).run(self.tasks)
        self.assertEqual((root / "answer.txt").read_text(), "15")
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(all(64 <= call["max_tokens"] < 512 for call in self.calls))
        self.assertEqual(report.budget.spent_units, 1200)

    async def test_forbidden_reviewer_is_checked_before_solver_http(self):
        root = self.task("unknown", "Give the result of adding eleven and four.")
        base = self.runner(models={"LLM_MODEL": MODELS[1], "LLM_REVIEW_MODEL": MODELS[3]})
        config = replace(
            base.config,
            attempt_limits=replace(base.config.attempt_limits, allowed_models=(MODELS[1],)),
        )
        components = composition.build_competition(
            base.ledger,
            config,
            transport=base.components.gateway._transport,
            environ={**ENV, "LLM_MODEL": MODELS[1], "LLM_REVIEW_MODEL": MODELS[3]},
        )
        staging = self.root / "restricted-staging"
        staging.mkdir()
        runner = AgentRunner(
            config=config, ledger=base.ledger, components=components, staging_parent=staging
        )
        self.addAsyncCleanup(runner.aclose)
        report = await runner.run(self.tasks)
        self.assertEqual(self.calls, [])
        self.assertEqual(report.budget.spent_units, 0)
        self.assertFalse((root / "answer.txt").exists())

    def runner(
        self,
        *,
        handler=None,
        usage=True,
        error=None,
        limit=100_000,
        models=None,
        output_cap=512,
        attempt_cost=None,
    ):
        config = RunConfig(
            total_budget_units=limit,
            deadline_monotonic=asyncio.get_running_loop().time() + 180,
            accounting_mode="best_effort",
            mode="portfolio",
            attempt_limits=AttemptLimits(
                max_cost_units=attempt_cost or limit,
                max_llm_calls=3,
                max_steps=8,
                max_duration_s=15,
                max_attempts_per_task=2,
                allowed_models=MODELS,
                tool_limits=ToolLimits(
                    timeout_s=2,
                    max_read_bytes=1_048_576,
                    max_output_bytes=1_048_576,
                    max_write_bytes=1000,
                    max_process_memory_bytes=100_000_000,
                    max_process_count=1,
                ),
            ),
        )
        ledger = BudgetLedger(limit_units=limit, deadline_monotonic=config.deadline_monotonic)

        async def deliver(request):
            wire = json.loads(request.content)
            self.calls.append(wire)
            self.assertGreater(ledger.snapshot().held_units, 0)
            payload = json.loads(wire["messages"][-1]["content"])
            if handler is not None:
                value = handler(wire, payload)
                if isinstance(value, httpx.Response):
                    return value
            else:
                value = direct("15") if payload["role"] == "solve" else review()
            if error == "timeout":
                raise httpx.ReadTimeout("mock timeout")
            if isinstance(error, int):
                return httpx.Response(
                    error,
                    json={"error": {"code": "quota_exhausted" if error == 402 else str(error)}},
                )
            response = {
                "choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]
            }
            if usage:
                response["usage"] = (
                    usage
                    if isinstance(usage, dict)
                    else {
                        "prompt_tokens": 120,
                        "completion_tokens": 48,
                        "total_tokens": 168,
                        "completion_tokens_details": {"reasoning_tokens": 0},
                    }
                )
            return httpx.Response(200, json=response)

        env = {**ENV, "LLM_OUTPUT_CAP": str(output_cap), **(models or {})}
        with patch(
            "nsu_budget_agent.llm.compatibility.load_compatibility",
            side_effect=AssertionError("competition must not load attestation"),
        ):
            components = composition.build_competition(
                ledger, config, transport=httpx.MockTransport(deliver), environ=env
            )
        staging = self.root / ("competition-staging-" + str(len(list(self.root.iterdir()))))
        staging.mkdir()
        runner = AgentRunner(
            config=config, ledger=ledger, components=components, staging_parent=staging
        )
        self.addAsyncCleanup(runner.aclose)
        return runner

    async def test_unknown_instruction_uses_unverified_qwen_roles_and_publishes(self):
        root = self.task("unknown", "Give the result of adding eleven and four.")
        runner = self.runner(models={"LLM_MODEL": MODELS[1], "LLM_REVIEW_MODEL": MODELS[3]})
        report = await runner.run(self.tasks)
        self.assertEqual((root / "answer.txt").read_text(), "15")
        self.assertEqual([c["model"] for c in self.calls], [MODELS[1], MODELS[3]])
        self.assertEqual(report.budget.spent_units, 1200)
        self.assertEqual(report.budget.held_units, 0)
        self.assertTrue(runner.components.gateway.competition_mode)
        self.assertTrue(
            all(
                not p.input_bound_verified and not p.total_output_cap_verified
                for p in runner.components.gateway._profiles.values()
            )
        )

    async def test_all_four_permitted_models_can_be_configured(self):
        root = self.task("affine", "Solve for x: 3*x+2=14. Return only the value of x.")
        for model in MODELS:
            runner = self.runner(models={"LLM_MODEL": model}, handler=lambda *_: direct("4"))
            await runner.run(self.tasks)
            self.assertEqual(self.calls[-1]["model"], model)
            self.assertEqual((root / "answer.txt").read_text(), "4")
            (root / "answer.txt").unlink()
        with self.assertRaises(ContractError):
            self.runner(models={"LLM_MODEL": "not-permitted"})

    async def test_unknown_usage_holds_estimate_stops_paid_and_keeps_free_answer(self):
        free = self.task("free", "Compute 2+3.", reward=1)
        self.task("high", "Give the result of adding eleven and four.", reward=90)
        self.task("other", "Give the result of adding eleven and four.", reward=50)
        runner = self.runner(usage=False)
        report = await runner.run(self.tasks)
        self.assertEqual((free / "answer.txt").read_text(), "5")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(report.budget.spent_units, 0)
        self.assertGreater(report.budget.held_units, 0)
        self.assertTrue(report.budget.blocked)

    async def test_ambiguous_reasoning_usage_retains_held_without_settlement(self):
        root = self.task("unknown", "Give the result of adding eleven and four.")
        runner = self.runner(
            usage={
                "prompt_tokens": 120,
                "completion_tokens": 48,
                "completion_tokens_details": {"reasoning_tokens": 12},
            }
        )
        report = await runner.run(self.tasks)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse((root / "answer.txt").exists())
        self.assertEqual(report.budget.spent_units, 0)
        self.assertGreater(report.budget.held_units, 0)
        self.assertTrue(report.budget.blocked)

    async def test_observed_output_cap_breach_is_accounted_and_stops_paid_routes(self):
        root = self.task("unknown", "Give the result of adding eleven and four.")
        runner = self.runner(
            usage={
                "prompt_tokens": 120,
                "completion_tokens": 600,
                "total_tokens": 720,
                "completion_tokens_details": {"reasoning_tokens": 0},
            }
        )
        report = await runner.run(self.tasks)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse((root / "answer.txt").exists())
        self.assertEqual(report.budget.spent_units, 6120)
        self.assertTrue(report.budget.blocked)

    async def test_publication_and_fsync_finish_before_later_failure(self):
        first = self.task("first", "Give the result of adding eleven and four.", reward=90)
        self.task(
            "second", "Give the result of adding eleven and four for this other task.", reward=50
        )
        original_fsync = os.fsync
        for failure in ("timeout", 402, 429, 500):
            self.calls.clear()

            def handler(wire, payload):
                if "other task" in payload["instruction"]:
                    self.assertEqual((first / "answer.txt").read_bytes(), b"15")
                    self.assertGreater(fsync.call_count, 0)
                    if failure == "timeout":
                        raise httpx.ReadTimeout("lost response")
                    return httpx.Response(
                        failure,
                        json={
                            "error": {"code": "quota_exhausted" if failure == 402 else str(failure)}
                        },
                    )
                return direct("15") if payload["role"] == "solve" else review()

            with (
                self.subTest(failure=failure),
                patch(
                    "nsu_budget_agent.submission.publisher.os.fsync", wraps=original_fsync
                ) as fsync,
            ):
                report = await self.runner(handler=handler).run(self.tasks)
            self.assertEqual((first / "answer.txt").read_bytes(), b"15")
            self.assertEqual(len(self.calls), 3)
            self.assertEqual(report.budget.spent_units, 1200)
            self.assertGreater(report.budget.held_units, 0)
            (first / "answer.txt").unlink()

    async def test_many_tasks_free_pass_reward_order_and_shared_budget(self):
        for i in range(8):
            self.task(f"free{i}", "Compute 2+3.", reward=1)
        for reward in (10, 100, 60, 40):
            self.task(
                f"paid{reward}",
                f"Give the result of adding eleven and four for reward {reward}.",
                reward=reward,
            )

        def handler(wire, payload):
            self.assertTrue(
                all((self.tasks / f"free{i}" / "answer.txt").read_text() == "5" for i in range(8))
            )
            return direct("15") if payload["role"] == "solve" else review()

        report = await self.runner(handler=handler).run(self.tasks)
        self.assertEqual(len(report.receipts), 12)
        self.assertEqual(
            [a.task_id for a in report.attempts if a.solver_id == "generic"],
            ["paid100", "paid60", "paid40", "paid10"],
        )
        self.assertEqual(report.budget.spent_units, 4800)

    async def test_output_cap_shrinks_to_available_attempt_budget(self):
        root = self.task("affine", "Solve for x: 3*x+2=14. Return only the value of x.")
        runner = self.runner(limit=6500, output_cap=2048, handler=lambda *_: direct("4"))
        report = await runner.run(self.tasks)
        self.assertEqual((root / "answer.txt").read_text(), "4")
        self.assertLess(self.calls[0]["max_tokens"], 2048)
        self.assertGreaterEqual(self.calls[0]["max_tokens"], 64)
        self.assertLessEqual(report.budget.spent_units + report.budget.held_units, 6500)

    async def test_cannot_afford_mandatory_review_never_spends_solver_http(self):
        root = self.task("unknown", "Give the result of adding eleven and four.")
        report = await self.runner(limit=6500).run(self.tasks)
        self.assertEqual(self.calls, [])
        self.assertFalse((root / "answer.txt").exists())
        self.assertEqual(report.budget.spent_units + report.budget.held_units, 0)

    async def test_process_killed_in_next_http_keeps_committed_answers_without_finally(self):
        first = self.task("accepted", "Give the result of adding eleven and four.", reward=90)
        free = self.task("free", "Compute 2+3.", reward=1)
        other = self.task(
            "crash", "Give the result of adding eleven and four for the crash task.", reward=50
        )
        marker = self.root / "second-http-started"
        env = {k: v for k, v in os.environ.items() if not k.startswith("LLM_")}
        env.update(
            ENV, BUDGET="100000", PYTHONPATH="src;.", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"
        )
        if os.name != "nt":
            env["PYTHONPATH"] = "src:."
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "tests.integration.a72_crash_worker",
            str(self.tasks),
            str(marker),
            str(self.staging_parent()),
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:

            async def wait_for_http():
                while not marker.exists():
                    if process.returncode is not None:
                        self.fail(f"worker exited before second HTTP: {process.returncode}")
                    await asyncio.sleep(0.05)

            await asyncio.wait_for(wait_for_http(), 60)
            self.assertEqual((first / "answer.txt").read_bytes(), b"15")
            process.kill()  # SIGKILL / TerminateProcess; no exception/finally guarantee.
            await asyncio.wait_for(process.wait(), 10)
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual((first / "answer.txt").read_bytes(), b"15")
            self.assertEqual((free / "answer.txt").read_bytes(), b"5")
            self.assertFalse((other / "answer.txt").exists())
            self.assertFalse(marker.with_suffix(".finally").exists())
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    def staging_parent(self):
        path = self.root / "crash-staging"
        path.mkdir()
        return path
