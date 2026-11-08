"""Varied public-input regressions through the real SDK; no network or gold input."""

import copy
import unittest

from tests.integration import test_a71_router as harness
from tests.integration.test_a71_router import review


class ReliabilityTests(unittest.IsolatedAsyncioTestCase):
    setUpClass = classmethod(harness.RouterPipelineTests.setUpClass.__func__)
    setUp = harness.RouterPipelineTests.setUp
    task = harness.RouterPipelineTests.task
    runner = harness.RouterPipelineTests.runner

    async def test_math_function_call_is_unavailable_without_review(self):
        root = self.task(
            "math",
            "What is the smallest positive integer divisible by 4, 6, and 10 with no remainder? Output just that integer.",
        )
        response = {
            "kind": "tool_plan",
            "tool_name": "local_math",
            "arguments": {"expression": "gcd(4, 6, 10)"},
        }
        await self.runner([response, review()]).run(self.tasks)
        self.assertFalse((root / "answer.txt").exists())
        self.assertEqual(len(self.calls), 1)

    async def test_math_numeric_expression_is_independently_checked(self):
        root = self.task(
            "math",
            "What is the smallest positive integer divisible by 4, 6, and 10 with no remainder? Output just that integer.",
        )
        response = {
            "kind": "tool_plan",
            "tool_name": "local_math",
            "arguments": {"expression": "3 * 4 * 5"},
        }
        report = await self.runner([response]).run(self.tasks)
        self.assertEqual((root / "answer.txt").read_text(), "60")
        self.assertTrue(report.decision_traces[0].independent_semantics)
        self.assertEqual(len(self.calls), 1)

    async def test_nested_lcm_is_independently_bound_to_question(self):
        root = self.task(
            "math",
            "What is the smallest positive integer divisible by 4, 6, and 10 with no remainder? Output just that integer.",
        )
        response = {
            "kind": "tool_plan",
            "tool_name": "local_math",
            "arguments": {"expression": "lcm(4, lcm(6, 10))"},
        }
        report = await self.runner([response]).run(self.tasks)
        self.assertEqual((root / "answer.txt").read_text(), "60")
        self.assertTrue(report.decision_traces[0].independent_semantics)
        self.assertEqual(len(self.calls), 1)

    async def test_lcm_wrong_operands_cannot_be_rescued_by_reviewer(self):
        root = self.task(
            "math",
            "What is the smallest positive integer divisible by 4, 6, and 10 with no remainder? Output just that integer.",
        )
        response = {
            "kind": "tool_plan",
            "tool_name": "local_math",
            "arguments": {"expression": "lcm(4, 6)"},
        }
        await self.runner([response, review()]).run(self.tasks)
        self.assertFalse((root / "answer.txt").exists())
        self.assertEqual(len(self.calls), 1)

    async def search_case(self, index, mutation=None):
        directory = f"profiles{index}"
        person, employer, parent = (
            f"{name} {chr(65 + index)}" for name in ("Lena", "Lab", "Group")
        )
        year = str(1980 + index)
        files = {
            f"{directory}/z.md": f"# {person}\nEmployer: {employer}\n",
            f"{directory}/b.md": f"# {employer}\nParent organization: {parent}\n",
            f"{directory}/a.md": f"# {parent}\nFounded: {year}\n",
        }
        instruction = (
            f"Using the files in workspace/{directory}/, identify {person}'s employer and then its parent organization. "
            "What year was that parent organization founded? Reply with the four-digit year. "
            f"Use only the Markdown documents provided under workspace/{directory}/. Do not use the internet."
        )
        root = self.task("relation", instruction, files=files, kind="search", hint="exact_str")
        args = {
            "paths": list(reversed(sorted(files))),
            "entity": person,
            "chain": ["Employer", "Parent organization"],
            "field": "Founded",
        }
        if mutation == "shortcut":
            args.update(paths=[f"{directory}/a.md"], entity=parent, chain=[])
        elif mutation == "intermediate":
            args.update(chain=["Employer"], field="Parent organization")
        elif mutation == "filename_entity":
            args["entity"] = f"{directory}/z.md"
        elif mutation == "wrong_relation":
            args["chain"] = ["Employer", "Owner"]
        elif mutation == "missing_source":
            args["paths"].remove(f"{directory}/b.md")
        elif mutation == "escape":
            args["paths"] = ["../hidden.md"]
        if index % 2 and mutation is None:
            args["paths"] = ["workspace/" + path for path in args["paths"]]
        response = {"kind": "tool_plan", "tool_name": "lookup_markdown", "arguments": args}

        def inspect_request(wire):
            import json

            payload = json.loads(wire["messages"][-1]["content"])
            schema = payload["tool_schemas"]["lookup_markdown"]["properties"]
            self.assertEqual(set(schema["paths"]["items"]["enum"]), set(files))
            self.assertEqual(schema["paths"]["minItems"], 3)
            self.assertEqual(schema["entity"], {"type": "string"})
            return response

        # A forged accept must never rescue a bad chain.
        runner = self.runner([inspect_request, review()])
        report = await runner.run(self.tasks)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(report.budget.held_units, 0)
        if mutation and mutation != "missing_source":
            self.assertFalse((root / "answer.txt").exists())
        else:
            self.assertEqual((root / "answer.txt").read_text(), year)
            self.assertTrue(report.decision_traces[0].independent_semantics)

    async def month_case(self, first, omit=False):
        path = f"period{first}.csv"
        root = self.task(
            "period",
            f"Using workspace/{path}, add up units for the north-region bolt orders "
            f"that are marked paid and whose month is {first}, {first + 1}, or {first + 2}. "
            "Do not include any other status or month. Give the exact integer total.",
            files={
                path: "product,region,status,month,units,price\n"
                f"bolt,north,paid,{first},4,2\nbolt,north,paid,{first + 2},6,3\n"
                "bolt,north,paid,12,99,2\nbolt,north,pending,3,40,2\n"
            },
            kind="data_analysis",
        )
        filters = [
            {"column": c, "op": "eq", "value": v}
            for c, v in (("region", "north"), ("product", "bolt"), ("status", "paid"))
        ]
        if not omit:
            filters += [
                {"column": "month", "op": "ge", "value": str(first)},
                {"column": "month", "op": "le", "value": str(first + 2)},
            ]
        response = {
            "kind": "tool_plan",
            "tool_name": "query_csv",
            "arguments": {
                "path": path,
                "aggregate": "sum",
                "columns": ["units"],
                "filters": filters,
            },
        }
        report = await self.runner([response, review()]).run(self.tasks)
        self.assertEqual(len(self.calls), 1)
        if omit:
            self.assertFalse((root / "answer.txt").exists())
        else:
            self.assertEqual((root / "answer.txt").read_text(), "10")
            self.assertTrue(report.decision_traces[0].independent_semantics)

    async def csv_case(self, index, mutation=None):
        path, product, region = f"records{index}.csv", f"item{index}", f"zone{index}"
        units, price = index + 2, index + 3
        text = (
            "product,region,status,units,price\n"
            f"{product},{region},paid,{units},{price}\n"
            f"{product},{region},refunded,100,20\n"
            f"{product},other,paid,100,20\n"
        )
        conditions = [f"product is {product}", f"region equals '{region}'", "status is paid"]
        if index % 2:
            conditions.reverse()
        instruction = (
            f"{'Using' if index % 2 else 'From'} workspace/{path}, "
            f"{'compute' if index % 3 else 'calculate'} the exact total revenue "
            "(units multiplied by price) from records where "
            + ", ".join(conditions[:-1])
            + ", and "
            + conditions[-1]
            + ". Return one integer."
        )
        root = self.task("data", instruction, files={path: text}, kind="data_analysis")
        args = {
            "path": path,
            "aggregate": "product_sum",
            "columns": ["units", "price"],
            "filters": [
                {"column": c, "op": "eq", "value": v}
                for c, v in (("status", "paid"), ("region", region), ("product", product))
            ],
        }
        if mutation == "missing_filter":
            args["filters"].pop(0)
        elif mutation == "wrong_column":
            args["columns"] = ["quantity", "price"]
        elif mutation == "wrong_aggregate":
            args.update(aggregate="count_rows", columns=[])
        elif mutation == "wrong_value":
            args["filters"][0]["value"] = "refunded"
        elif mutation == "bad_operator":
            args["filters"][0]["op"] = "ne"
        elif mutation == "numeric_value":
            args["filters"][0]["value"] = 1
        elif mutation == "escape":
            args["path"] = "../private.csv"
        response = {"kind": "tool_plan", "tool_name": "query_csv", "arguments": copy.deepcopy(args)}

        def inspect_request(wire):
            import json

            payload = json.loads(wire["messages"][-1]["content"])
            schema = payload["tool_schemas"]["query_csv"]
            self.assertEqual(schema["properties"]["path"]["enum"], [path])
            self.assertIn("units", schema["description"])
            self.assertNotIn(product, schema["description"])
            return response

        runner = self.runner([inspect_request, review()])
        report = await runner.run(self.tasks)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(report.budget.held_units, 0)
        if mutation:
            self.assertFalse((root / "answer.txt").exists())
        else:
            self.assertEqual((root / "answer.txt").read_text(), str(units * price))
            self.assertTrue(report.decision_traces[0].independent_semantics)


def scenario(family, index, mutation=None):
    async def test(self):
        await getattr(self, family + "_case")(index, mutation)

    return test


for _family in ("search", "csv"):
    for _index in range(12):
        setattr(ReliabilityTests, f"test_{_family}_varied_{_index:02}", scenario(_family, _index))
for _first in (2, 6):
    for _omit in (False, True):

        def month_scenario(first, omit):
            async def test(self):
                await self.month_case(first, omit)

            return test

        setattr(
            ReliabilityTests, f"test_month_{_first}_omit_{_omit}", month_scenario(_first, _omit)
        )
for _family, _mutations in (
    (
        "search",
        (
            "shortcut",
            "intermediate",
            "filename_entity",
            "wrong_relation",
            "missing_source",
            "escape",
        ),
    ),
    (
        "csv",
        (
            "missing_filter",
            "wrong_column",
            "wrong_aggregate",
            "wrong_value",
            "bad_operator",
            "numeric_value",
            "escape",
        ),
    ),
):
    for _index, _mutation in enumerate(_mutations):
        setattr(
            ReliabilityTests,
            f"test_{_family}_reject_{_mutation}",
            scenario(_family, _index, _mutation),
        )
