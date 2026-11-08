"""Independent operation binding and numerical checks cover the existing grammar."""

import time
import unittest
from dataclasses import replace
from pathlib import Path, PurePosixPath

from nsu_budget_agent.contracts.models import ArtifactRef, TaskSpec, ToolCall
from nsu_budget_agent.solvers.plans import plan
from nsu_budget_agent.validation.certification import independent_math
from nsu_budget_agent.validation.instruction import instruction_matches


class IndependentCertificateTests(unittest.TestCase):
    def task(self, instruction):
        return TaskSpec(
            task_id="certificate",
            instruction=instruction,
            reward=10,
            task_root=Path.cwd(),
            work_root=Path.cwd(),
            artifacts=tuple(
                ArtifactRef(path=PurePosixPath(p), size_bytes=0, sha256="0" * 64)
                for p in ("data.csv", "kb/a.md")
            ),
        )

    def test_math_signs_floor_modulo_power_decimal_and_stock(self):
        for expression, answer in (
            ("-5//3", "-2"),
            ("-5%3", "1"),
            ("5%-3", "-1"),
            ("0**0", "1"),
            ("2**12", "4096"),
            ("0.1+0.2", "0.3"),
            ("1/8", "0.125"),
        ):
            task = self.task("Compute " + expression + ".")
            call = plan(task).call
            self.assertTrue(instruction_matches(task, call), expression)
            self.assertEqual(independent_math(call.arguments, time.monotonic() + 5), answer)
        task = self.task(
            "A warehouse starts with 100 boxes. It ships 2 boxes per day for 3 days. Then the stock is split equally into 3 warehouses and it keeps one share (rounded down). How many boxes are left?"
        )
        call = plan(task).call
        self.assertTrue(instruction_matches(task, call))
        self.assertEqual(independent_math(call.arguments, time.monotonic() + 5), "31")

    def test_independent_binding_checks_every_existing_data_question_shape(self):
        instructions = (
            "In data.csv, what is the sum of amount where region equals 'west'?",
            "In data.csv, how many rows where region equals 'west'?",
            "The file data.csv has columns order_id,product,region (3 rows). How many orders of product 'book' came from region 'west'?",
            "The file data.csv has columns units,product,month,region (3 rows). How many units of product 'book' were sold in region 'west' during months 2 through 4 inclusive?",
            "The file data.csv has columns units,price,product,month (3 rows). What is the total revenue (units multiplied by price) for product 'book' in months 2 through 4 inclusive?",
            "In which year was Alice born?",
            "In which city is the headquarters of Alice's employer?",
            "In which year was Alice's employer founded?",
            "The directory kb contains a small knowledge base of markdown files. In which year was Alice's mentor born?",
        )
        for instruction in instructions:
            task = self.task(instruction)
            call = plan(task).call
            self.assertTrue(instruction_matches(task, call), instruction)
            changed = dict(call.arguments)
            if call.tool_name == "query_csv":
                changed["filters"] = []
            else:
                changed["entity"] = "Bob"
            self.assertFalse(
                instruction_matches(task, ToolCall(tool_name=call.tool_name, arguments=changed))
            )
            self.assertFalse(
                instruction_matches(
                    replace(task, instruction=instruction + " Except cancelled entries."), call
                )
            )
