"""Regressions for the eight independently reproduced A7.1 review defects."""

import asyncio
import json
import unittest
from dataclasses import replace

from tests.integration import test_a71_router as harness
from tests.integration.test_a71_router import CSV, PARTIAL, direct, review

from nsu_budget_agent.solvers.decision import State


class ReviewFixTests(unittest.IsolatedAsyncioTestCase):
    setUpClass = classmethod(harness.RouterPipelineTests.setUpClass.__func__)
    setUp = harness.RouterPipelineTests.setUp
    task = harness.RouterPipelineTests.task
    runner = harness.RouterPipelineTests.runner
    states = harness.RouterPipelineTests.states

    async def test_negation_alternative_and_qualification_never_certified_as_entity(self):
        for index, entity in enumerate(
            ("Alice, not Bob,", "Alice or Bob", "Alice from London", "Alice Rather Than Bob")
        ):
            root = self.task(
                str(index),
                f"In which year was {entity} born?",
                kind="search",
                files={
                    "a.md": "# Alice\nBorn: 1980\n",
                    "b.md": "# Bob\nBorn: 1985\n",
                    "c.md": f"# {entity}\nBorn: 2001\n",
                },
            )
        report = await self.runner(
            [{"kind": "decline", "reason": "unsupported qualification"}] * 4
        ).run(self.tasks)
        self.assertEqual(report.receipts, ())
        for trace in report.decision_traces:
            self.assertNotIn(State.CERTIFIED, [s for s, _ in trace.events])
        self.assertFalse((root / "answer.txt").exists())

    async def test_rejected_30_cannot_return_as_direct_answer_and_get_accepted(self):
        root = self.task(
            "csv",
            PARTIAL + " Use only rows with status 'paid'.",
            files={"sales.csv": CSV},
            kind="data_analysis",
        )
        report = await self.runner(
            [review("reject", "replan", ["status paid"]), direct("30", ["sales.csv"]), review()]
        ).run(self.tasks)
        self.assertFalse((root / "answer.txt").exists())
        self.assertEqual(report.receipts, ())
        self.assertEqual(len(self.calls), 2)

    async def test_unneeded_credentials_never_enter_either_prompt(self):
        root = self.task(
            "arithmetic",
            "Give the result of adding eleven and four.",
            files={"credentials.txt": "token=TEST_SECRET_SENTINEL_451\n"},
        )
        report = await self.runner([direct("15"), review()]).run(self.tasks)
        self.assertEqual((root / "answer.txt").read_text(), "15")
        self.assertEqual(len(report.receipts), 1)
        for wire in self.calls:
            self.assertNotIn("TEST_SECRET_SENTINEL_451", json.dumps(wire))
            self.assertNotIn("credentials.txt", json.dumps(wire))

    async def test_secret_explicit_reference_is_denied_before_http(self):
        self.task(
            "secret",
            "Give the token in credentials.txt.",
            files={"credentials.txt": "token=TEST_SECRET_SENTINEL_451\n"},
            hint="exact_str",
        )
        report = await self.runner().run(self.tasks)
        self.assertEqual(report.receipts, ())
        self.assertEqual(self.calls, [])

    async def test_sensitive_content_in_an_explicit_normal_source_is_not_sent(self):
        self.task(
            "source",
            "Total paid west revenue in sales.csv.",
            files={"sales.csv": CSV + "api_key=TEST_SECRET_SENTINEL_451\n"},
        )
        report = await self.runner().run(self.tasks)
        self.assertEqual(report.receipts, ())
        self.assertEqual(self.calls, [])

    async def test_forged_free_csv_sum_999_is_not_certified_or_published(self):
        root = self.task("csv", PARTIAL, files={"sales.csv": CSV})
        report = await self.runner(forge=True).run(self.tasks)
        self.assertFalse((root / "answer.txt").exists())
        self.assertNotIn(State.CERTIFIED, self.states(report, "csv"))

    async def test_closed_free_grammar_does_not_inherit_llm_source_count_limit(self):
        root = self.task(
            "many",
            "In which year was Alice born?",
            kind="search",
            files={
                "a.md": "# Alice\nBorn: 1980\n",
                "b.md": "# Bob\nBorn: 1985\n",
                "c.md": "# Carol\nBorn: 1986\n",
                "d.md": "# David\nBorn: 1987\n",
            },
        )
        report = await self.runner().run(self.tasks)
        self.assertEqual((root / "answer.txt").read_text(), "1980")
        self.assertEqual(self.calls, [])
        self.assertIn(State.CERTIFIED, self.states(report, "many"))

    async def test_forged_free_math_and_markdown_are_independently_rejected(self):
        self.task("math", "Compute 2+3.")
        self.task(
            "markdown",
            "In which year was Alice born?",
            kind="search",
            files={"alice.md": "# Alice\nBorn: 1980\n"},
        )

        def forge(components):
            original = components.tools.execute

            async def execute(call_id, call, ctx):
                result = await original(call_id, call, ctx)
                if call.tool_name in ("local_math", "lookup_markdown") and result.status == "ok":
                    return replace(result, data={**result.data, "answer": "999"})
                return result

            components.tools.execute = execute
            return components

        report = await self.runner(mutate=forge).run(self.tasks)
        self.assertEqual(report.receipts, ())
        for trace in report.decision_traces:
            self.assertNotIn(State.CERTIFIED, [s for s, _ in trace.events])

    async def test_budget_24000_keeps_highest_reward_before_cheaper_tasks(self):
        for name, reward in (("high", 90), ("middle", 50), ("low", 1)):
            self.task(name, "Give the result of adding eleven and four.", reward=reward)
        report = await self.runner([direct("15"), review()] * 3, limit=24_000).run(self.tasks)
        self.assertEqual(len(report.receipts), 3)
        self.assertEqual(
            [a.task_id for a in report.attempts if a.solver_id == "generic"],
            ["high", "middle", "low"],
        )
        self.assertEqual(report.budget.spent_units, 3600)

    async def test_repeated_run_same_id_changed_instruction_is_fresh(self):
        root = self.task("same", "Compute 2+3.")
        runner = self.runner()
        first = await runner.run(self.tasks)
        self.assertEqual((root / "answer.txt").read_text(), "5")
        (root / "answer.txt").unlink()
        data = json.loads((root / "task.json").read_text())
        data["instruction"] = "Compute 3+3."
        (root / "task.json").write_text(json.dumps(data))
        second = await runner.run(self.tasks)
        self.assertEqual((root / "answer.txt").read_text(), "6")
        self.assertEqual(len(first.attempts), 1)
        self.assertEqual(len(second.attempts), 1)
        self.assertEqual(second.workflow_invocations, 1)

    async def test_overlapping_runs_are_explicitly_rejected(self):
        from nsu_budget_agent.contracts.models import ContractError

        self.task("slow", "Give the result of adding eleven and four.")
        runner = self.runner([direct("15")], error="slow", duration=0.3)
        first = asyncio.create_task(runner.run(self.tasks))
        await asyncio.sleep(0)
        try:
            with self.assertRaises(ContractError):
                await runner.run(self.tasks)
        finally:
            await first

    async def test_required_review_preflight_spends_no_single_call(self):
        self.task("unknown", "Give the result of adding eleven and four.")
        report = await self.runner([direct("15")], llm_calls=1).run(self.tasks)
        self.assertEqual(self.calls, [])
        self.assertEqual(report.budget.spent_units, 0)

    async def test_unaffordable_reservation_is_deferred_without_consuming_paid_route(self):
        self.task("pending", "Give the result of adding eleven and four.")
        report = await self.runner(limit=100).run(self.tasks)
        self.assertEqual(report.progress[0].status, "deferred")
        self.assertEqual(report.progress[0].attempted_solvers, ("local_router",))
        self.assertEqual(self.calls, [])
        self.assertEqual(report.decision_traces[0].state, State.UNSOLVED)

    async def test_larger_followup_bound_defers_instead_of_exhausting_reward(self):
        from nsu_budget_agent.solvers.composition import build_with_router

        high = self.task(
            "high",
            PARTIAL + " Use only rows with status 'paid'.",
            files={"sales.csv": CSV},
            reward=90,
        )
        low = self.task("low", "Give the result of adding eleven and four.", reward=1)

        def factory(ledger, config, *, compatibility, transport, environ):
            def bound(request, body):
                data = json.loads(request.messages[-1].content)
                proposal = data.get("proposal")
                if (
                    data["role"] == "review"
                    and proposal
                    and proposal["operation"] == "query_csv"
                    and len(proposal["arguments"]["filters"]) == 2
                ):
                    return 24880  # 30000 Units, beyond the initial fair allocation.
                return 4096

            return build_with_router(
                ledger,
                config,
                compatibility=replace(compatibility, input_bound=bound),
                transport=transport,
                environ=environ,
            )

        report = await self.runner(
            [
                review("reject", "replan", ["status paid"]),
                harness.csv_plan(),
                direct("15"),
                review(),
            ],
            factory=factory,
            limit=50_000,
        ).run(self.tasks)
        progress = {p.task_id: p for p in report.progress}
        self.assertEqual(progress["high"].status, "deferred")
        self.assertFalse((high / "answer.txt").exists())
        self.assertEqual((low / "answer.txt").read_text(), "15")
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(report.budget.spent_units, 2400)

    async def test_parallel_runners_sharing_components_have_independent_run_state(self):
        from nsu_budget_agent.orchestration.runtime import AgentRunner

        self.task("same", "Compute 2+3.")
        first_root = self.tasks
        first = self.runner()
        self.tasks = self.root / "other-tasks"
        self.tasks.mkdir()
        self.task("same", "Compute 3+3.")
        staging = self.root / "other-staging"
        staging.mkdir()
        second = AgentRunner(
            config=first.config,
            ledger=first.ledger,
            components=first.components,
            staging_parent=staging,
        )
        self.addAsyncCleanup(second.aclose)
        self.assertIsNot(first.components.decision_store, second.components.decision_store)
        a, b = await asyncio.gather(first.run(first_root), second.run(self.tasks))
        self.assertEqual(len(a.receipts), 1)
        self.assertEqual(len(b.receipts), 1)
        self.assertEqual((first_root / "same" / "answer.txt").read_text(), "5")
        self.assertEqual((self.tasks / "same" / "answer.txt").read_text(), "6")

    async def test_v1_solvers_are_stateless_and_stop_keeps_terminal_semantics(self):
        from nsu_budget_agent.contracts.models import StopAttempt
        from nsu_budget_agent.solvers import decision

        self.assertFalse(hasattr(decision, "ContinueDecision"))
        self.task("local", "Compute 2+3.")
        runner = self.runner()
        for solver in runner.components.registry.all_solvers():
            self.assertFalse(hasattr(solver, "store"))
        # A normal v1 consumer sees only ordinary DTOs, never a hidden yield.
        from nsu_budget_agent.contracts.models import AttemptBudget
        from nsu_budget_agent.contracts.ports import SolverContext
        from nsu_budget_agent.task_io import DirectoryTaskLoader

        task = (await DirectoryTaskLoader().load(self.tasks)).tasks[0]
        budget = AttemptBudget(
            max_cost_units=0,
            max_llm_calls=0,
            max_steps=8,
            deadline_monotonic=runner.config.deadline_monotonic,
            allowed_models=(),
            tool_limits=runner.config.attempt_limits.tool_limits,
        )

        class Denied:
            async def complete(self, request):
                raise AssertionError("v1 stop must not call LLM")

        ctx = SolverContext(
            task=task,
            attempt_id="plain",
            step_index=0,
            budget=budget,
            tools=runner.components.tools.catalog(),
            llm=Denied(),
        )
        generic = runner.components.registry.get("generic")
        a = await generic.step(ctx, ())
        b = await generic.step(ctx, ())
        self.assertIs(type(a), StopAttempt)
        self.assertEqual(a, b)
