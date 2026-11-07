import os
import unittest
from decimal import Inexact, InvalidOperation, Overflow, Rounded, Underflow, localcontext
from unittest.mock import patch

from nsu_budget_agent.budget import budget_from_env, parse_budget
from nsu_budget_agent.contracts.models import ContractError


class BudgetConfigTests(unittest.TestCase):
    def test_explicit_units_and_conservative_decimal_conversion(self):
        for text, expected in (
            ("0", 0),
            ("0e999999999999999999", 0),
            ("0.09", 0),
            ("0.1", 1),
            ("72", 720),
            ("72.1999", 721),
            (" 1e2 ", 1000),
            ("0.29", 2),
        ):
            with self.subTest(value=text):
                self.assertEqual(parse_budget(text, unit="competition_tokens"), expected)
        for text in ("720", "720.0", "7.2e2"):
            self.assertEqual(parse_budget(text, unit="units_x10"), 720)
        self.assertEqual(budget_from_env(unit="competition_tokens", environ={"BUDGET": "72"}), 720)
        self.assertEqual(budget_from_env(unit="units_x10", environ={"BUDGET": "72"}), 72)
        with patch.dict(os.environ, {"BUDGET": "72"}):
            self.assertEqual(budget_from_env(unit="competition_tokens"), 720)

    def test_decimal_precision_does_not_round_away_units(self):
        with localcontext() as context:
            context.prec = 2
            self.assertEqual(
                parse_budget("123456789012345678901234567890.19", unit="competition_tokens"),
                1234567890123456789012345678901,
            )
            self.assertEqual(context.prec, 2)

    def test_conversion_ignores_caller_decimal_context(self):
        with localcontext() as context:
            context.prec = 2
            context.Emax = 2
            context.Emin = -2
            for signal in (Inexact, Overflow, Rounded, Underflow):
                context.traps[signal] = True
            context.traps[InvalidOperation] = False
            original_flags = context.flags.copy()
            self.assertEqual(
                budget_from_env(unit="competition_tokens", environ={"BUDGET": "100.5"}), 1005
            )
            self.assertEqual(parse_budget("0.000001", unit="competition_tokens"), 0)
            self.assertEqual(parse_budget("1005.0", unit="units_x10"), 1005)
            with self.assertRaises(ContractError):
                parse_budget("0.5", unit="units_x10")
            with self.assertRaises(ContractError):
                parse_budget("invalid", unit="competition_tokens")
            self.assertEqual(context.flags, original_flags)

    def test_invalid_or_missing_budget_fails_without_a_default(self):
        for unit in ("competition_tokens", "units_x10"):
            for value in (
                "",
                " ",
                "-0.1",
                "NaN",
                "sNaN",
                "Infinity",
                "-Infinity",
                "abc",
                True,
                1,
                0.1,
                None,
            ):
                with self.subTest(value=value, unit=unit), self.assertRaises(ContractError):
                    parse_budget(value, unit=unit)
        with self.assertRaises(ContractError):
            parse_budget("0.5", unit="units_x10")
        with self.assertRaises(ContractError):
            parse_budget("1", unit="unknown")
        with self.assertRaises(ContractError):
            budget_from_env(unit="competition_tokens", environ={})
        with self.assertRaises(TypeError):
            budget_from_env(environ={"BUDGET": "1"})
