"""Explicit bounded mathematical operation; arbitrary Python remains unavailable."""

import unittest

from nsu_budget_agent.tools.calculations import arithmetic
from nsu_budget_agent.validation.certification import independent_math


class BoundedLCMTests(unittest.TestCase):
    def test_positive_values_nested_calls_and_arithmetic(self):
        for expression, answer in (
            ("lcm(2, 3, 7)", "42"),
            ("lcm(5,lcm(9,6))", "90"),
            ("lcm(11,13)+2", "145"),
            ("lcm(12,18,24)", "72"),
        ):
            with self.subTest(expression=expression):
                self.assertEqual(arithmetic({"expression": expression}, lambda: None), answer)
                self.assertEqual(independent_math({"expression": expression}, 10**20), answer)

    def test_invalid_operands_and_calls_remain_rejected(self):
        expressions = (
            "lcm(0,2)",
            "lcm(-1,2)",
            "lcm(1.5,2)",
            "lcm(1000001,2)",
            "lcm(2)",
            "lcm(" + ",".join(["2"] * 17) + ")",
            "lcm(a=2,b=3)",
            "math.lcm(2,3)",
            "gcd(2,3)",
            "__import__('os')",
            "lcm(*[2,3])",
        )
        for expression in expressions:
            for calculate in (
                lambda args: arithmetic(args, lambda: None),
                lambda args: independent_math(args, 10**20),
            ):
                with self.subTest(expression=expression, calculate=calculate):
                    with self.assertRaises(ValueError):
                        calculate({"expression": expression})
