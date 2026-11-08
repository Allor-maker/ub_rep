"""Public instructions, synthetic evidence and independent oracles; no live API."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from tests.integration import test_a71_router as harness
from tests.integration.test_a71_router import direct, review

from nsu_budget_agent.orchestration.decision import State
from nsu_budget_agent.solvers.composition import build
from nsu_budget_agent.solvers.plans import explicit_missing_data
from nsu_budget_agent.task_io import DirectoryTaskLoader
from nsu_budget_agent.validation import LocalValidator

MATH = (
    "A warehouse starts with 908 boxes. It ships an unspecified number of boxes per day "
    "for 2 days. Exactly how many boxes remain? The daily shipment count is not included in "
    "the available data. Work only inside the current task workspace. Write ONLY the final "
    "answer to answer.txt next to task.json, with no explanation."
)
DATA = (
    "The file sales.csv in workspace/ has columns order_id,region,product,month,units,price "
    "(100 rows). How much sales tax was actually collected for product 'gizmo' in months 2 "
    "through 4 inclusive? No tax rate or tax column is provided. Use python3 to compute an "
    "exact result from the available data. Work only inside the current task workspace. "
    "Write ONLY the final answer to answer.txt next to task.json, with no explanation."
)
SEARCH = (
    "The directory kb/ in workspace/ contains a small knowledge base of Markdown files. "
    "In which year did Nikolai Berg's employer acquire its first foreign subsidiary? Use the "
    "available files to find it; do not infer unsupported facts. Work only inside the current "
    "task workspace. Write ONLY the final answer to answer.txt next to task.json, with no explanation."
)


def profiles(*, acquisition=None):
    company = "# Dune Robotics\nHeadquarters: Lima\nFounded: 1972\n"
    if acquisition is not None:
        company += f"First foreign subsidiary acquisition year: {acquisition}\n"
    files = {
        "kb/person_nikolai_berg.md": "# Nikolai Berg\nBorn: 1968\nEmployer: Dune Robotics\nMentor: Elena Haddad\n",
        "kb/company_dune_robotics.md": company,
    }
    files.update({f"kb/other_{i}.md": f"# Other {i}\nFounded: 2000\n" for i in range(22)})
    return files


class LiveBenchmarkRegressionTests(unittest.IsolatedAsyncioTestCase):
    setUpClass = classmethod(harness.RouterPipelineTests.setUpClass.__func__)
    setUp = harness.RouterPipelineTests.setUp
    task = harness.RouterPipelineTests.task
    runner = harness.RouterPipelineTests.runner
    states = harness.RouterPipelineTests.states

    async def test_closed_missing_data_guard_does_not_match_modified_instructions(self):
        self.task("math", MATH)
        self.task(
            "data", DATA, kind="data_analysis", files={"sales.csv": "product,price\ngizmo,10\n"}
        )
        loaded = (await DirectoryTaskLoader().load(self.tasks)).tasks
        for task in loaded:
            self.assertIsNotNone(explicit_missing_data(task))
            self.assertIsNone(
                explicit_missing_data(
                    replace(task, instruction=task.instruction + " Further information follows.")
                )
            )

    async def test_offline_comparison_uses_declines_without_live_calls(self):
        import nsu_budget_agent.orchestration.decision as decision

        original_plan = decision.plan

        def former_plan(task):
            if task.instruction == SEARCH:
                raise ValueError("former unsupported search")
            return original_plan(task)

        metrics = []
        for old in (True, False):
            self.tasks = self.root / ("old" if old else "new")
            self.tasks.mkdir()
            self.task("math_0004", MATH, reward=13)
            self.task(
                "data_analysis_0004",
                DATA,
                reward=3,
                kind="data_analysis",
                files={
                    "sales.csv": "order_id,region,product,month,units,price\n"
                    + "".join(f"{i},north,gizmo,2,3,10\n" for i in range(1, 101))
                },
            )
            self.task(
                "search_0004", SEARCH, reward=3, kind="search", hint="exact_str", files=profiles()
            )
            self.calls.clear()
            responses = [{"kind": "decline", "reason": "missing input"}] * 2
            if old:
                with (
                    patch.object(decision, "explicit_missing_data", return_value=None),
                    patch(
                        "nsu_budget_agent.solvers.decision.explicit_missing_data", return_value=None
                    ),
                    patch.object(decision, "plan", side_effect=former_plan),
                ):
                    report = await self.runner(responses).run(self.tasks)
            else:
                report = await self.runner(responses).run(self.tasks)
            metrics.append((len(report.receipts), len(self.calls), report.budget.spent_units))
        self.assertEqual(metrics, [(0, 2, 1200), (0, 0, 0)])

    async def test_missing_warehouse_quantity_never_publishes_or_spends(self):
        root = self.task("math_0004", MATH, kind="math")
        report = await self.runner([direct("900"), review()]).run(self.tasks)
        self.assertFalse((root / "answer.txt").exists())
        self.assertEqual(len(self.calls), 0)
        self.assertEqual(report.budget.spent_units, 0)

    async def test_missing_tax_never_publishes_revenue_as_tax(self):
        csv = "order_id,region,product,month,units,price\n" + "".join(
            f"{i},north,gizmo,2,3,10\n" for i in range(1, 101)
        )
        root = self.task("data_analysis_0004", DATA, kind="data_analysis", files={"sales.csv": csv})
        report = await self.runner([direct("3000", ["sales.csv"]), review()]).run(self.tasks)
        self.assertFalse((root / "answer.txt").exists())
        self.assertEqual(len(self.calls), 0)
        self.assertEqual(report.budget.spent_units, 0)

    async def test_known_warehouse_quantity_still_solves_without_http(self):
        text = (
            "A warehouse starts with 908 boxes. It ships 3 boxes per day for 2 days. "
            "How many boxes are left? Work only inside the current task workspace. "
            "Write ONLY the final answer to answer.txt next to task.json, with no explanation."
        )
        root = self.task("known_math", text, kind="math")
        report = await self.runner().run(self.tasks)
        self.assertEqual((root / "answer.txt").read_text(), "902")
        self.assertIn(State.CERTIFIED, self.states(report, "known_math"))
        self.assertEqual(len(self.calls), 0)

    async def test_known_revenue_still_solves_without_http(self):
        text = DATA.replace(
            "How much sales tax was actually collected for product 'gizmo' in months 2 through 4 inclusive? "
            "No tax rate or tax column is provided.",
            "What is the total revenue (units multiplied by price) for product 'gizmo' in months 2 through 4 inclusive?",
        )
        csv = "order_id,region,product,month,units,price\n" + "".join(
            f"{i},north,gizmo,2,3,10\n" for i in range(1, 101)
        )
        root = self.task("known_data", text, kind="data_analysis", files={"sales.csv": csv})
        report = await self.runner().run(self.tasks)
        self.assertEqual((root / "answer.txt").read_text(), "3000")
        self.assertIn(State.CERTIFIED, self.states(report, "known_data"))
        self.assertEqual(len(self.calls), 0)

    async def test_search_missing_fact_refuses_without_http(self):
        root = self.task("search_0004", SEARCH, kind="search", hint="exact_str", files=profiles())
        report = await self.runner().run(self.tasks)
        self.assertFalse((root / "answer.txt").exists())
        self.assertEqual(len(self.calls), 0)
        self.assertNotIn(State.CERTIFIED, self.states(report, "search_0004"))

    async def test_search_explicit_acquisition_year_is_independently_certified(self):
        root = self.task(
            "search_0004",
            SEARCH,
            kind="search",
            hint="exact_str",
            files=profiles(acquisition="2011"),
        )
        report = await self.runner().run(self.tasks)
        self.assertEqual((root / "answer.txt").read_text(), "2011")
        self.assertEqual(len(self.calls), 0)
        self.assertIn(State.CERTIFIED, self.states(report, "search_0004"))

    async def test_search_forged_answer_is_not_certified(self):
        root = self.task(
            "search_0004",
            SEARCH,
            kind="search",
            hint="exact_str",
            files=profiles(acquisition="2011"),
        )

        def forge(components):
            original = components.tools.execute

            async def execute(call_id, call, ctx):
                result = await original(call_id, call, ctx)
                if call.tool_name == "lookup_markdown" and result.status == "ok":
                    return replace(result, data={**result.data, "answer": "999"})
                return result

            components.tools.execute = execute
            return components

        report = await self.runner(mutate=forge).run(self.tasks)
        self.assertFalse((root / "answer.txt").exists())
        self.assertNotIn(State.CERTIFIED, self.states(report, "search_0004"))

    async def test_offline_validator_independently_rejects_forged_search(self):
        root = self.task(
            "search_0004",
            SEARCH,
            kind="search",
            hint="exact_str",
            files=profiles(acquisition="2011"),
        )

        def forge(components):
            original = components.tools.execute

            async def execute(call_id, call, ctx):
                result = await original(call_id, call, ctx)
                if call.tool_name == "lookup_markdown" and result.status == "ok":
                    return replace(result, data={**result.data, "answer": "999"})
                return result

            components.tools.execute = execute
            return replace(components, validator=LocalValidator(components.tools))

        report = await self.runner(factory=build, mutate=forge).run(self.tasks)
        self.assertFalse((root / "answer.txt").exists())
        self.assertEqual(report.receipts, ())

    async def test_offline_search_publishes_independently_checked_year(self):
        root = self.task(
            "search_0004",
            SEARCH,
            kind="search",
            hint="exact_str",
            files=profiles(acquisition="2011"),
        )
        report = await self.runner(factory=build).run(self.tasks)
        self.assertEqual((root / "answer.txt").read_text(), "2011")
        self.assertEqual(len(report.receipts), 1)

    async def test_search_bad_year_and_forged_evidence_are_rejected(self):
        for kind in ("bad_year", "bad_evidence"):
            self.tasks = self.root / kind
            self.tasks.mkdir()
            root = self.task(
                "search_0004",
                SEARCH,
                kind="search",
                hint="exact_str",
                files=profiles(acquisition="2011 or 2012" if kind == "bad_year" else "2011"),
            )

            def forge(components):
                original = components.tools.execute

                async def execute(call_id, call, ctx):
                    result = await original(call_id, call, ctx)
                    if (
                        kind == "bad_evidence"
                        and call.tool_name == "lookup_markdown"
                        and result.status == "ok"
                    ):
                        return replace(
                            result, data={**result.data, "evidence_paths": ["kb/other_0.md"]}
                        )
                    return result

                components.tools.execute = execute
                return components

            report = await self.runner(mutate=forge).run(self.tasks)
            self.assertFalse((root / "answer.txt").exists())
            self.assertNotIn(State.CERTIFIED, self.states(report, "search_0004"))

    async def test_search_qualification_not_silently_ignored(self):
        root = self.task(
            "search_0004",
            SEARCH.replace("Nikolai Berg's", "Nikolai Berg, not Alice,'s"),
            kind="search",
            hint="exact_str",
            files=profiles(acquisition="2011"),
        )
        report = await self.runner().run(self.tasks)
        self.assertFalse((root / "answer.txt").exists())
        self.assertNotIn(State.CERTIFIED, self.states(report, "search_0004"))

    async def test_diagnostic_codes_distinguish_model_and_router(self):
        cases = (
            ("invalid", "not json", "invalid_json"),
            ("decline", {"kind": "decline", "reason": "missing"}, "model_declined"),
            (
                "tool",
                {"kind": "tool_plan", "tool_name": "shell", "arguments": {}},
                "unsupported_operation",
            ),
        )
        for label, response, expected in cases:
            self.tasks = self.root / label
            self.tasks.mkdir()
            self.task(label, "Give the result of adding eleven and four.")
            self.calls.clear()
            report = await self.runner([response]).run(self.tasks)
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(report.decision_traces[0].events[-1][1], expected)
