"""Generalized families and changed parameters; no grader or network access."""

import unittest
from pathlib import PurePosixPath

from tests.integration import test_a71_router as harness
from tests.integration.test_a71_router import direct

from nsu_budget_agent.solvers.sources import selected_sources
from nsu_budget_agent.task_io import DirectoryTaskLoader
from nsu_budget_agent.tools.calculations import markdown_lookup
from nsu_budget_agent.validation.data_plan import _markdown_answer, data_question
from nsu_budget_agent.validation.llm import question


class QualityTests(unittest.IsolatedAsyncioTestCase):
    setUpClass = classmethod(harness.RouterPipelineTests.setUpClass.__func__)
    setUp = harness.RouterPipelineTests.setUp
    task = harness.RouterPipelineTests.task
    runner = harness.RouterPipelineTests.runner

    async def test_workspace_prefix_selects_only_referenced_public_sources(self):
        self.task(
            "csv",
            "Use workspace/orders.csv to answer.",
            files={"orders.csv": "a\n1\n", "other.csv": "a\n2\n"},
        )
        self.task(
            "kb",
            "Using workspace/kb/, find the answer.",
            files={
                "kb/a.md": "# A\nBorn: 2001\n",
                "kb/token.txt": "token=SECRET\n",
                "other/b.md": "# B\nBorn: 1990\n",
            },
        )
        tasks = {t.task_id: t for t in (await DirectoryTaskLoader().load(self.tasks)).tasks}
        self.assertEqual(
            [a.path.as_posix() for a in selected_sources(tasks["csv"])], ["orders.csv"]
        )
        self.assertEqual([a.path.as_posix() for a in selected_sources(tasks["kb"])], ["kb/a.md"])

    async def test_numeric_profile_property_is_data(self):
        args = {
            "paths": ["org.md"],
            "entity": "Council",
            "chain": [],
            "field": "2022 summit host city",
        }
        text = "# Council\n2022 summit host city: Kyoto\n2021 summit host city: Oslo\n"
        self.assertEqual(markdown_lookup({"org.md": text}, args, lambda: None)[0], "Kyoto")
        self.assertEqual(
            _markdown_answer({PurePosixPath("org.md"): text}, args, 10**20)[0], "Kyoto"
        )

    async def test_independent_data_direct_answer_needs_no_reviewer(self):
        root = self.task(
            "data",
            "From sales.csv, total the amount values for rows whose status is 'paid'.",
            files={"sales.csv": "amount,status\n10,paid\n20,pending\n"},
            kind="data_analysis",
        )
        report = await self.runner([direct("10", ["sales.csv"])]).run(self.tasks)
        self.assertEqual((root / "answer.txt").read_text(), "10")
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(report.decision_traces[0].independent_semantics)

    async def test_conflicting_clarifications_are_not_certified(self):
        main = (
            "From workspace/orders.csv, calculate the exact total revenue (units multiplied by price) "
            "from records where product is gadget, region is north, and status is paid. "
        )
        self.task(
            "conflict",
            main + "Count orders rather than summing units, and do not count cancelled rows.",
            files={"orders.csv": "product,region,status,units,price\ngadget,north,paid,3,7\n"},
        )
        self.task(
            "extra",
            "Find the sum of the first five prime numbers, starting with 2. Except exclude 3. Give the final integer only.",
        )
        tasks = {t.task_id: t for t in (await DirectoryTaskLoader().load(self.tasks)).tasks}
        self.assertIsNone(data_question(tasks["conflict"]))
        self.assertIsNone(question(tasks["extra"]))

    async def test_general_math_families_use_one_verified_model_answer(self):
        cases = (
            (
                "Find the sum of the first five prime numbers, starting with 2. Give the final integer only.",
                "28",
            ),
            (
                "What is the smallest positive integer divisible by 4, 6, and 10 with no remainder? Output just that integer.",
                "60",
            ),
            (
                "A small theatre has 4 rows. The first row has 3 seats, and each subsequent row has 2 more seats than the preceding row. How many seats are there in total across all 4 rows? Answer with an integer.",
                "24",
            ),
            (
                "A lock code has exactly 2 different digits chosen from 0, 1, 2, and 3. The order matters and a code may begin with zero. How many distinct codes are possible? Answer as an integer.",
                "12",
            ),
        )
        for i, (instruction, answer) in enumerate(cases):
            self.tasks = self.root / f"math{i}"
            self.tasks.mkdir()
            root = self.task("math", instruction)
            self.calls.clear()
            report = await self.runner([direct(answer)]).run(self.tasks)
            self.assertEqual((root / "answer.txt").read_text(), answer)
            self.assertEqual(len(self.calls), 1)
            self.assertTrue(report.decision_traces[0].independent_semantics)

    async def test_general_csv_conditions_are_independently_verified(self):
        instruction = (
            "From workspace/orders.csv, calculate the exact total revenue (units multiplied by price) "
            "from records where product is gadget, region is north, and status is paid. "
            "Exclude pending, cancelled or refunded records even if they have matching product and region. Return one integer."
        )
        text = "product,region,status,month,units,price\ngadget,north,paid,2,3,7\ngadget,north,refunded,2,9,7\ngadget,south,paid,2,8,7\n"
        root = self.task("csv", instruction, files={"orders.csv": text}, kind="data_analysis")
        report = await self.runner([direct("21", ["orders.csv"])]).run(self.tasks)
        self.assertEqual((root / "answer.txt").read_text(), "21")
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(report.decision_traces[0].independent_semantics)

    async def test_relation_chain_direct_answer_is_verified(self):
        instruction = (
            "Using the files in workspace/kb/, identify Mina Lee's employer and then its parent organization. "
            "What year was that parent organization founded? Reply with the four-digit year. "
            "Use only the Markdown documents provided under workspace/kb/. Do not use the internet."
        )
        files = {
            "kb/person.md": "# Mina Lee\nEmployer: Alpha Lab\n",
            "kb/company.md": "# Alpha Lab\nParent organization: Beta Group\n",
            "kb/parent.md": "# Beta Group\nFounded: 2001\n",
        }
        root = self.task("search", instruction, files=files, kind="search", hint="exact_str")
        report = await self.runner([direct("2001", sorted(files))]).run(self.tasks)
        self.assertEqual((root / "answer.txt").read_text(), "2001")
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(report.decision_traces[0].independent_semantics)

    async def test_wrong_csv_status_total_cannot_be_certified(self):
        root = self.task(
            "wrong_csv",
            "From workspace/orders.csv, calculate the exact total revenue (units multiplied by price) "
            "from records where product is gadget, region is north, and status is paid. Return one integer.",
            files={
                "orders.csv": "product,region,status,units,price\ngadget,north,paid,3,7\ngadget,north,refunded,9,7\n"
            },
            kind="data_analysis",
        )
        await self.runner([direct("84", ["orders.csv"])]).run(self.tasks)
        self.assertFalse((root / "answer.txt").exists())
        self.assertEqual(len(self.calls), 1)

    async def test_gateway_events_contain_usage_and_no_prompt(self):
        self.task(
            "events",
            "Find the sum of the first five prime numbers, starting with 2. Give the final integer only.",
        )
        runner = self.runner([direct("28")])
        await runner.run(self.tasks)
        from dataclasses import asdict

        events = [asdict(e) for e in runner.components.gateway.events]
        self.assertEqual(sum(e["phase"] == "dispatched" for e in events), 1)
        self.assertTrue(any(e["input_tokens"] is not None for e in events))
        self.assertFalse(any("prompt" in e or "content" in e or "api_key" in e for e in events))

    async def test_allowed_tool_plan_aliases_and_commutative_filters(self):
        instruction = (
            "From workspace/orders.csv, calculate the exact total revenue (units multiplied by price) "
            "from records where product is gadget, region is north, and status is paid. Return one integer."
        )
        root = self.task(
            "csv",
            instruction,
            files={
                "orders.csv": "product,region,status,units,price\ngadget,north,paid,3,7\ngadget,north,cancelled,8,7\n"
            },
            kind="data_analysis",
        )
        response = {
            "kind": "tool_plan",
            "tool_name": "query_csv",
            "arguments": {
                "path": "workspace/orders.csv",
                "aggregate": "product_sum",
                "columns": ["units", "price"],
                "filters": [
                    {"column": "status", "op": "eq", "value": "paid"},
                    {"column": "region", "op": "eq", "value": "north"},
                    {"column": "product", "op": "eq", "value": "gadget"},
                ],
            },
        }
        report = await self.runner([response]).run(self.tasks)
        self.assertEqual((root / "answer.txt").read_text(), "21")
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(report.decision_traces[0].independent_semantics)

    async def test_wrong_math_answer_is_blocked_without_review_or_retry(self):
        root = self.task(
            "wrong",
            "Find the sum of the first five prime numbers, starting with 2. Give the final integer only.",
        )
        report = await self.runner([direct("29")]).run(self.tasks)
        self.assertFalse((root / "answer.txt").exists())
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(report.decision_traces[0].independent_semantics)

    async def test_local_math_expression_is_checked_against_full_question(self):
        root = self.task(
            "primes",
            "Find the sum of the first five prime numbers, starting with 2. Give the final integer only.",
        )
        response = {
            "kind": "tool_plan",
            "tool_name": "local_math",
            "arguments": {"expression": "2+3+5+7+11"},
        }
        report = await self.runner([response]).run(self.tasks)
        self.assertEqual((root / "answer.txt").read_text(), "28")
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(report.decision_traces[0].independent_semantics)

    async def test_row_count_month_units_and_revenue_variants(self):
        csv = "product,region,status,month,units,price\nwidget,east,shipped,4,9,2\nwidget,east,paid,4,3,2\nwidget,east,paid,5,5,2\nwidget,east,paid,8,7,2\n"
        cases = (
            (
                "In workspace/orders.csv, how many ORDER ROWS for product widget in the east region have status shipped? Count orders rather than summing units, and do not count paid, pending, cancelled, or refunded rows. Output just the count.",
                "1",
            ),
            (
                "Using workspace/orders.csv, add up units for the east-region widget orders that are marked paid and whose month is 3, 4, or 5. Do not include any other status or month. Give the exact integer total.",
                "8",
            ),
            (
                "The workspace/orders.csv file lists units, price and status. What is the revenue from east-region widget orders in months 3 through 5 inclusive that have status paid? Refunded orders must not count. Compute revenue as units times price and return its exact integer sum.",
                "16",
            ),
        )
        for i, (instruction, answer) in enumerate(cases):
            self.tasks = self.root / f"rows{i}"
            self.tasks.mkdir()
            root = self.task("csv", instruction, files={"orders.csv": csv}, kind="data_analysis")
            self.calls.clear()
            report = await self.runner([direct(answer, ["workspace/orders.csv"])]).run(self.tasks)
            self.assertEqual((root / "answer.txt").read_text(), answer)
            self.assertEqual(len(self.calls), 1)
            self.assertTrue(report.decision_traces[0].independent_semantics)
