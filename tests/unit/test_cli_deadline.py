"""CLI runtime deadline defaults and explicit overrides."""

import os
import unittest
from unittest.mock import AsyncMock, patch

from nsu_budget_agent.orchestration.cli import _reason_code, main


class CliDeadlineTests(unittest.TestCase):
    def test_reason_codes_keep_known_refusals_without_dynamic_text(self):
        self.assertEqual(
            _reason_code("review rejected or unresolved after bounded strategy"),
            "review_rejected_or_unresolved",
        )
        self.assertEqual(_reason_code("invalid_json"), "invalid_json")
        self.assertIsNone(_reason_code("private arbitrary candidate text"))

    def test_runtime_deadline_default_and_override(self):
        for arguments, expected in (([], 3300), (["--timeout-s", "2.5"], 2.5)):
            with self.subTest(arguments=arguments):
                run = AsyncMock(return_value=0)
                with patch.dict(os.environ), patch("nsu_budget_agent.orchestration.cli._run", run):
                    self.assertEqual(
                        main(["--tasks-dir", "tasks", *arguments], factory=lambda *_: None), 0
                    )
                self.assertEqual(run.await_args.args[0].timeout_s, expected)

    def test_attempt_timeout_is_separate_and_configurable(self):
        run = AsyncMock(return_value=0)
        with patch("nsu_budget_agent.orchestration.cli._run", run):
            self.assertEqual(
                main(
                    ["--tasks-dir", "tasks", "--timeout-s", "900", "--attempt-timeout-s", "75"],
                    factory=lambda *_: None,
                ),
                0,
            )
        self.assertEqual(run.await_args.args[0].attempt_timeout_s, 75)
        self.assertEqual(run.await_args.args[0].timeout_s, 900)
